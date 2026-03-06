# ♛ Chess AI Engine

A fully self-contained chess engine and graphical game written in Python.
Play **Human (White) vs AI (Black)** on a rendered Pygame board, or open
`ui/chess_web.html` in any browser for a zero-install web version.

---

## Quick Start

```bash
# 1 — Install dependencies
pip install pygame numpy scikit-learn

# 2 — Enter the project folder
cd chess_ai_engine

# 3 — Run
python main.py
```

For the **web version** — just double-click `ui/chess_web.html`. No Python needed.

---

## Controls (Pygame)

| Key / Action        | Effect                          |
|---------------------|---------------------------------|
| Left-click a piece  | Select it (legal moves shown)   |
| Left-click a square | Move the selected piece         |
| **R**               | Restart game                    |
| **F**               | Flip board                      |
| **ESC**             | Quit                            |

---

## Project Structure

```
chess_ai_engine/
│
├── main.py              ← Game loop & state machine
├── board.py             ← Board representation & rules
├── move_generator.py    ← Legal move generation
├── evaluation.py        ← Position scoring
├── minimax.py           ← Baseline minimax (reference)
├── alpha_beta.py        ← Main AI search engine
├── zobrist_hash.py      ← Hashing & transposition table
├── opening_book.py      ← Opening book loader
├── ml_model.py          ← ML evaluation (RandomForest)
│
├── ui/
│   ├── pygame_board.py  ← Desktop graphical UI
│   └── chess_web.html   ← Standalone browser version
│
└── data/
    ├── opening_book.pkl ← Auto-generated on first run
    └── ml_model.pkl     ← Saved ML model (if trained)
```

---

## Algorithms Used

### 1. Board Representation — `board.py`

The board is stored as an **8×8 NumPy int8 array**.

```
Positive integers = White pieces
Negative integers = Black pieces
Zero              = Empty square

PAWN=1  KNIGHT=2  BISHOP=3  ROOK=4  QUEEN=5  KING=6
```

Every move is stored as a `Move` object containing:
- Source and destination squares
- The piece moved and piece captured
- Flags for castling, en passant, and promotion
- **Saved state** for the undo operation (en passant square, castling rights, halfmove clock)

This allows **make_move / undo_move** to be fully reversible without copying the board.

---

### 2. Legal Move Generation — `move_generator.py`

Moves are generated in two stages:

**Stage 1 — Pseudo-legal moves** (fast, may leave king in check):
- Pawns: single push, double push from starting rank, diagonal captures, en passant
- Knights: all 8 L-shape jumps
- Sliding pieces (Bishop / Rook / Queen): ray casting — extend in each direction until blocked
- King: 8 adjacent squares
- Castling: checks empty squares and attacked squares along the path

**Stage 2 — Legality filter**:
Each pseudo-legal move is made, and the position is checked for whether the moving side's king is in check. If yes, the move is discarded.

**Attack detection** (`is_square_attacked`) uses the same ray-casting idea in reverse — stand on the target square and shoot rays outward to see if any attacking piece is on the ray.

---

### 3. Minimax Search — `minimax.py`

The **minimax algorithm** is the foundation of game-tree search.

```
White (maximiser) wants the highest score.
Black (minimiser) wants the lowest score.

At each node:
  - If maximising: return max(children)
  - If minimising: return min(children)
  - At depth 0: return static evaluation
```

Minimax guarantees finding the optimal move given perfect evaluation, but explores **every** node in the tree — too slow for real play. It is kept in this project as a readable reference implementation.

Complexity: **O(b^d)** where b ≈ 30 (branching factor) and d = depth.
At depth 5: ~30^5 = 24 million nodes.

---

### 4. Alpha-Beta Pruning — `alpha_beta.py`  *(main engine)*

Alpha-beta is a minimax optimisation that **prunes branches** that cannot affect the final result.

```
α (alpha) = best score the maximiser is already guaranteed
β (beta)  = best score the minimiser is already guaranteed

If a node's score ≥ β  →  the minimiser won't choose this branch → PRUNE (β cutoff)
If a node's score ≤ α  →  the maximiser won't choose this branch → PRUNE (α cutoff)
```

In the best case, alpha-beta reduces the effective branching factor from b to √b,
making **depth 10 as fast as minimax depth 5**.

#### Additional Enhancements

**Iterative Deepening**
Search at depth 1, then 2, then 3, … up to `AI_DEPTH`.
Each shallower search guides move ordering for the next depth.
Also enables time management — stop when time limit is reached.

**Negamax Formulation**
Instead of separate max/min players, use `score = -search(opponent)`.
Cleaner code; both sides use the same function.

**Move Ordering**
Alpha-beta prunes more when good moves are searched first.
Order: `TT move → captures (MVV-LVA) → killer moves → history heuristic → quiet moves`

- **MVV-LVA** (Most Valuable Victim – Least Valuable Attacker): prefer capturing a queen with a pawn over capturing a pawn with a queen.
- **Killer moves**: two quiet moves per ply that recently caused a beta cutoff.
- **History heuristic**: bonus accumulated each time a quiet move raises alpha.

**Quiescence Search**
At depth 0 (leaf nodes), instead of evaluating immediately, continue searching **captures only** until a quiet position is reached. Prevents the "horizon effect" where the engine misses a capture one move beyond its depth.

**Null Move Pruning**
Temporarily skip a turn (pass). If the resulting position is still good enough to cause a beta cutoff, the real position must also be very good → prune. Saves significant time in non-critical positions.

**Late Move Reduction (LMR)**
Moves searched late in the list (after the first 4) are likely bad.
Search them at reduced depth first. If the reduced-depth result is interesting,
re-search at full depth. Otherwise skip.

**Transposition Table**
Positions reached by different move orders are often identical.
Cache the result with `(hash, depth, score, flag, best_move)`.
Three flag types:
- `EXACT` — the stored score is accurate
- `ALPHA` — score is an upper bound (failed low)
- `BETA`  — score is a lower bound (failed high / caused cutoff)

---

### 5. Zobrist Hashing — `zobrist_hash.py`

A **Zobrist hash** is a 64-bit integer that uniquely identifies a board position.

**Setup**: assign a random 64-bit number to each (piece, square) combination (12 piece types × 64 squares = 768 numbers), plus numbers for side to move, castling rights, and en passant file.

**Hash computation**:
```
hash = 0
for each piece on the board:
    hash ^= table[piece][square]
if black to move:
    hash ^= black_to_move_key
hash ^= castling_table[castling_rights]
if en_passant:
    hash ^= ep_table[ep_file]
```

**Incremental update**: instead of recomputing from scratch after each move, XOR out the pieces that moved and XOR in their new positions. This is O(1) per move.

The hash is used as the key into the **Transposition Table**.

---

### 6. Evaluation Function — `evaluation.py`

The static evaluator scores a position from White's perspective (positive = White better).

#### A. Material Score
Each piece is worth a fixed number of **centipawns** (1 pawn = 100 cp):

| Piece  | Value  |
|--------|--------|
| Pawn   | 100 cp |
| Knight | 320 cp |
| Bishop | 330 cp |
| Rook   | 500 cp |
| Queen  | 900 cp |
| King   | ∞      |

#### B. Piece-Square Tables (PST)
Each piece gets a **bonus or penalty** based on where it stands on the board.
Tables are hard-coded 8×8 arrays of bonuses in centipawns.

Examples:
- Knights score higher in the centre (d4/e4/d5/e5), lower on the edges
- Pawns score higher as they advance toward promotion
- King scores lower in the centre during the middlegame (safety), higher in the endgame (activity)

The table is mirrored vertically for Black.

#### C. Mobility
Count the number of pseudo-legal moves available to each side.
```
score += (white_moves - black_moves) × 4 cp
```
More moves = more options = better position.

#### D. King Safety
In the middlegame, reward pawns in front of the king (pawn shield) and penalise open files near the king.

#### E. Pawn Structure
- **Doubled pawns** (two pawns on same file): −20 cp each
- **Isolated pawns** (no friendly pawns on adjacent files): −15 cp each
- **Passed pawns** (no enemy pawns blocking or adjacent): +20 to +105 cp depending on how advanced

#### F. Bishop Pair
Having both bishops is worth +30 cp (more mobility in open positions).

---

### 7. Opening Book — `opening_book.py`

The engine plays pre-programmed moves for the first ~14 moves using a lookup table.

Openings included:
- Ruy Lopez (e4 e5 Nf3 Nc6 Bb5)
- Italian Game (e4 e5 Nf3 Nc6 Bc4)
- Scotch Game
- Sicilian Defense (e4 c5)
- French Defense (e4 e6)
- Caro-Kann (e4 c6)
- Scandinavian (e4 d5)
- Queen's Gambit (d4 d5 c4)
- Queen's Gambit Declined
- Slav Defense
- King's Indian Defense
- Nimzo-Indian
- English Opening
- Réti Opening

The book is stored as a dict `{FEN_key → [candidate_moves]}` and saved to `data/opening_book.pkl`.
A random candidate is chosen at each book position to add variety.

---

### 8. Machine Learning Evaluation — `ml_model.py`

An **optional** RandomForestRegressor that learns to evaluate positions.

**Feature vector** (46 features per position):
- Material difference per piece type (5)
- Raw piece counts per side (10)
- Total material + material balance (2)
- Pawn structure metrics: doubled, isolated, advancement (6)
- King safety: centrality, pawn shield count (4)
- Mobility estimate (2)
- Center control (2)
- Castling rights (2)
- Side to move (1)
- ... plus derived features

**Training flow**:
1. Play 1,000 random positions by simulating games with some randomness
2. Evaluate each position using the classical `Evaluator`
3. Train `RandomForestRegressor(n_estimators=100, max_depth=10)`
4. Validate on a held-out 20% split
5. Save model to `data/ml_model.pkl`

Enable it in `main.py` by setting `USE_ML_EVAL = True`.

---

### 9. Game State Machine — `main.py`

The game loop uses three explicit states to prevent race conditions between the main thread and the AI background thread:

```
STATE_HUMAN   → waiting for the player to click a move
     ↓  (player makes a move)
STATE_AI      → AI thread computing in background
     ↓  (thread finishes, result collected)
STATE_HUMAN   → repeat
     ↓  (no legal moves)
STATE_GAMEOVER
```

The AI receives a **snapshot copy** of the board (`board.copy()`) so the background thread cannot interfere with the main thread's board state.

---

### 10. Piece Rendering — `ui/pygame_board.py`

All 12 piece types (6 × White + 6 × Black) are drawn using **pure pygame polygon, circle, and rectangle calls** — no image files, no unicode fonts needed.

Each piece is composed of layered shapes drawn bottom-to-top:

| Piece  | Anatomy                                                     |
|--------|-------------------------------------------------------------|
| Pawn   | Wide flat base → thin neck stem → large circle head        |
| Rook   | Wide base → rectangular tower body → three battlement blocks|
| Knight | Wide base → angled neck trapezoid → horse-head polygon → muzzle box → eye dot |
| Bishop | Wide base → tapering lower body → narrow upper body → circle head → mitre spike → cross engraving |
| Queen  | Wide base → tapered body trapezoid → crown band → 5 orb balls → centre gem |
| King   | Wide base → tapered body trapezoid → crown band → vertical cross shaft → horizontal cross arm |

White pieces: **ivory fill** + dark brown outline
Black pieces: **near-black fill** + gold outline

Piece surfaces are cached after the first render (`_surface_cache`) so they are only drawn once.

---

## Configuration

Edit the top of `main.py`:

```python
HUMAN_COLOR      = WHITE    # play as WHITE or BLACK
AI_DEPTH         = 3        # search depth (3=fast, 5=strong, 7=very slow)
AI_TIME_LIMIT    = 5.0      # max seconds per AI move
USE_OPENING_BOOK = True     # use pre-programmed opening moves
USE_ML_EVAL      = False    # use RandomForest evaluation (slower first run)
```

---

## Performance

At `AI_DEPTH = 3` on a modern laptop:

| Metric      | Typical value     |
|-------------|-------------------|
| Nodes/sec   | 5,000 – 15,000    |
| Move time   | 0.1 – 0.5 seconds |
| TT hit rate | 15 – 30%          |

At `AI_DEPTH = 5`:

| Metric      | Typical value     |
|-------------|-------------------|
| Nodes/sec   | 5,000 – 15,000    |
| Move time   | 1 – 6 seconds     |
| TT hit rate | 25 – 45%          |

---

## Dependencies

| Library       | Used for                                |
|---------------|-----------------------------------------|
| `pygame`      | Window, rendering, mouse/keyboard input |
| `numpy`       | Board array, Zobrist random table       |
| `scikit-learn`| RandomForestRegressor (ML eval)         |
| `pickle`      | Save/load opening book and ML model     |

---

## License

MIT — free to use, modify, and distribute.
