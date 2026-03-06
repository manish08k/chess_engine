"""
alpha_beta.py - Alpha-Beta Pruning Search Engine

Advanced chess search using alpha-beta pruning to dramatically reduce
the number of nodes evaluated. Features include:
- Alpha-beta pruning with fail-soft framework
- Iterative deepening with time management
- Transposition table integration
- Move ordering (TT move, MVV-LVA, killer moves, history heuristic)
- Quiescence search to avoid horizon effect
- Null move pruning
- Check extensions
"""

import time
from typing import Optional, Tuple, List
from board import Board, Move, WHITE, BLACK, EMPTY
from evaluation import Evaluator
from move_generator import MoveGenerator
from zobrist_hash import ZobristHash, TranspositionTable, TT_EXACT, TT_ALPHA, TT_BETA

INF = 10_000_000
CHECKMATE_SCORE = 9_000_000
DRAW_SCORE = 0

# Killer move table size
MAX_DEPTH = 64


class AlphaBetaSearch:
    """
    Production-strength chess search engine using alpha-beta pruning
    with multiple heuristic enhancements.
    """

    def __init__(self, board: Board, max_depth: int = 5, time_limit: float = 5.0):
        self.board = board
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.evaluator = Evaluator()
        self.zobrist = ZobristHash()
        self.tt = TranspositionTable(size_mb=16)

        # Search statistics
        self.nodes_searched = 0
        self.tt_hits = 0
        self.start_time = 0.0

        # Move ordering heuristics
        # Killer moves: quiet moves that caused beta cutoff [ply][slot]
        self.killers: List[List[Optional[Move]]] = [[None, None] for _ in range(MAX_DEPTH)]
        # History heuristic: bonus for moves that improved alpha
        self.history: dict = {}

        self.best_move: Optional[Move] = None
        self.current_depth = 0

    # ------------------------------------------------------------------
    # Main Search Entry Point
    # ------------------------------------------------------------------

    def search(self) -> Optional[Move]:
        """
        Iterative deepening alpha-beta search.
        Returns the best move found within the time limit.
        """
        self.nodes_searched = 0
        self.tt_hits = 0
        self.start_time = time.time()
        self.best_move = None

        # Reset heuristic tables
        self.killers = [[None, None] for _ in range(MAX_DEPTH)]
        self.history = {}

        # Compute initial hash
        board_hash = self.zobrist.compute_hash(self.board)
        color = self.board.turn

        # Check for legal moves first
        legal_moves = MoveGenerator(self.board).generate_legal_moves(color)
        if not legal_moves:
            return None
        if len(legal_moves) == 1:
            return legal_moves[0]

        # Iterative deepening: search at increasing depths
        for depth in range(1, self.max_depth + 1):
            self.current_depth = depth
            score, move = self._root_search(depth, board_hash, color)

            elapsed = time.time() - self.start_time
            if move:
                self.best_move = move

            if elapsed >= self.time_limit:
                break

            # Mate found: no need to search deeper
            if abs(score) >= CHECKMATE_SCORE - MAX_DEPTH:
                break

        return self.best_move

    def _root_search(self, depth: int, board_hash, color: int) -> Tuple[int, Optional[Move]]:
        """
        Root node search: returns (best_score, best_move).
        Handles move ordering using TT best move.
        """
        alpha = -INF
        beta = INF
        best_score = -INF
        best_move = None

        mg = MoveGenerator(self.board)
        moves = mg.generate_legal_moves(color)

        # Get TT move for ordering
        _, tt_move = self.tt.lookup(board_hash, depth, alpha, beta)
        moves = mg.order_moves(moves, tt_move)

        for move in moves:
            if time.time() - self.start_time >= self.time_limit:
                break

            if self.board.make_move(move):
                score = -self._alphabeta(depth - 1, -beta, -alpha, 1, -color)
                self.board.undo_move(move)

                if score > best_score:
                    best_score = score
                    best_move = move
                    if score > alpha:
                        alpha = score

        # Store in TT
        flag = TT_EXACT
        self.tt.store(board_hash, depth, best_score, flag, best_move)

        return best_score, best_move

    # ------------------------------------------------------------------
    # Alpha-Beta with Enhancements
    # ------------------------------------------------------------------

    def _alphabeta(self, depth: int, alpha: int, beta: int,
                    ply: int, color: int) -> int:
        """
        Recursive alpha-beta search (negamax formulation).

        Args:
            depth: Remaining search depth
            alpha: Lower bound (maximizing player's best guaranteed score)
            beta:  Upper bound (minimizing player's best guaranteed score)
            ply:   Distance from root (used for killer moves)
            color: Current side to move

        Returns:
            Score from the current player's perspective (negamax)
        """
        self.nodes_searched += 1

        # Time check (every 4096 nodes to reduce overhead)
        if self.nodes_searched & 4095 == 0:
            if time.time() - self.start_time >= self.time_limit:
                return 0  # Abort search

        # Transposition table lookup
        board_hash = self.zobrist.compute_hash(self.board)
        tt_score, tt_move = self.tt.lookup(board_hash, depth, alpha, beta)
        if tt_score is not None:
            self.tt_hits += 1
            return tt_score

        # Quiescence search at leaf nodes
        if depth <= 0:
            return self._quiescence(alpha, beta, color, ply)

        # Check for drawn positions
        if self.board.is_draw_by_fifty_moves():
            return DRAW_SCORE
        if self.board.is_insufficient_material():
            return DRAW_SCORE

        # Generate moves
        mg = MoveGenerator(self.board)
        moves = mg.generate_legal_moves(color)

        # Terminal position
        if not moves:
            if self.board.is_in_check(color):
                return -(CHECKMATE_SCORE - ply)  # Prefer shorter mates
            return DRAW_SCORE  # Stalemate

        # Check extension: search one extra ply when in check
        in_check = self.board.is_in_check(color)
        if in_check:
            depth += 1

        # Null move pruning (skip our turn; if still good, prune)
        # Only in non-check positions with sufficient material
        null_reduction = 2
        if (not in_check and depth >= 3 and
                not self._is_endgame() and ply > 0):
            self.board.turn = -color  # Pass turn to opponent
            null_score = -self._alphabeta(
                depth - 1 - null_reduction, -beta, -beta + 1, ply + 1, -color
            )
            self.board.turn = color  # Restore turn
            if null_score >= beta:
                return beta  # Beta cutoff

        # Move ordering
        killers = self.killers[ply] if ply < MAX_DEPTH else [None, None]
        moves = self._order_moves_with_killers(moves, mg, tt_move, killers)

        best_score = -INF
        best_move = None
        original_alpha = alpha

        for i, move in enumerate(moves):
            if not self.board.make_move(move):
                continue

            # Late move reduction (LMR): reduce search for quiet late moves
            if (i >= 4 and depth >= 3 and not in_check and
                    move.captured == EMPTY and move.promotion == EMPTY and
                    not self.board.is_in_check(-color)):
                # Search with reduced depth
                score = -self._alphabeta(depth - 2, -alpha - 1, -alpha, ply + 1, -color)
                if score > alpha:
                    # Re-search with full depth
                    score = -self._alphabeta(depth - 1, -beta, -alpha, ply + 1, -color)
            else:
                score = -self._alphabeta(depth - 1, -beta, -alpha, ply + 1, -color)

            self.board.undo_move(move)

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score

                # Update history heuristic for quiet moves
                if move.captured == EMPTY:
                    key = (move.from_sq, move.to_sq)
                    self.history[key] = self.history.get(key, 0) + depth * depth

            if alpha >= beta:
                # Beta cutoff: update killer moves for quiet moves
                if move.captured == EMPTY and ply < MAX_DEPTH:
                    self._update_killers(move, ply)
                break

        # Store result in transposition table
        if best_score <= original_alpha:
            flag = TT_ALPHA
        elif best_score >= beta:
            flag = TT_BETA
        else:
            flag = TT_EXACT

        self.tt.store(board_hash, depth, best_score, flag, best_move)
        return best_score

    # ------------------------------------------------------------------
    # Quiescence Search
    # ------------------------------------------------------------------

    def _quiescence(self, alpha: int, beta: int, color: int, ply: int) -> int:
        """
        Quiescence search: continue searching captures to avoid horizon effect.
        Evaluates only capture moves until a "quiet" position is reached.
        """
        self.nodes_searched += 1

        # Stand-pat evaluation
        stand_pat = self.evaluator.evaluate_for_side(self.board, color)

        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        # Generate only capture moves
        mg = MoveGenerator(self.board)
        captures = [m for m in mg.generate_legal_moves(color) if m.captured != EMPTY]
        captures = mg.order_moves(captures)

        for move in captures:
            if self.board.make_move(move):
                score = -self._quiescence(-beta, -alpha, -color, ply + 1)
                self.board.undo_move(move)

                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score

        return alpha

    # ------------------------------------------------------------------
    # Move Ordering Helpers
    # ------------------------------------------------------------------

    def _order_moves_with_killers(self, moves: List[Move], mg: MoveGenerator,
                                    tt_move: Optional[Move],
                                    killers: List[Optional[Move]]) -> List[Move]:
        """Order moves: TT move > captures (MVV-LVA) > killers > history > quiet."""
        PIECE_VALUES = {1: 100, 2: 320, 3: 330, 4: 500, 5: 900, 6: 20000, 0: 0}

        def score(move: Move) -> int:
            if tt_move and move == tt_move:
                return 2_000_000

            if move.captured != EMPTY:
                victim = PIECE_VALUES.get(abs(move.captured), 0)
                attacker = PIECE_VALUES.get(abs(move.piece), 0)
                return 1_000_000 + victim * 10 - attacker

            if move.promotion != EMPTY:
                return 900_000

            # Killer move bonus
            if killers and any(k is not None and k == move for k in killers):
                return 800_000

            # History heuristic
            return self.history.get((move.from_sq, move.to_sq), 0)

        return sorted(moves, key=score, reverse=True)

    def _update_killers(self, move: Move, ply: int):
        """Update killer moves at this ply (keep 2 slots)."""
        k0 = self.killers[ply][0]
        if k0 is None or k0 != move:
            self.killers[ply][1] = self.killers[ply][0]
            self.killers[ply][0] = move

    def _is_endgame(self) -> bool:
        """Rough check for endgame phase (limited material)."""
        from board import QUEEN, ROOK
        queens = rooks = 0
        for r in range(8):
            for c in range(8):
                p = abs(int(self.board.squares[r][c]))
                if p == QUEEN:
                    queens += 1
                elif p == ROOK:
                    rooks += 1
        return queens == 0 or (queens == 2 and rooks <= 2)

    def get_stats(self) -> dict:
        """Return search statistics."""
        elapsed = time.time() - self.start_time
        nps = int(self.nodes_searched / elapsed) if elapsed > 0 else 0
        return {
            'nodes': self.nodes_searched,
            'tt_hits': self.tt_hits,
            'depth': self.current_depth,
            'time_ms': int(elapsed * 1000),
            'nps': nps,
            'best_move': str(self.best_move) if self.best_move else None,
        }
