"""
evaluation.py - Chess Position Evaluation

Evaluates chess positions using:
- Material counting with piece values
- Piece-square tables (positional bonuses)
- Mobility scoring
- King safety evaluation
- Pawn structure analysis
"""

import numpy as np
from typing import Dict, Tuple
from board import Board, WHITE, BLACK, EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING

# -----------------------------------------------------------------------
# Piece Base Values (centipawns)
# -----------------------------------------------------------------------
PIECE_VALUES: Dict[int, int] = {
    PAWN:   100,
    KNIGHT: 320,
    BISHOP: 330,
    ROOK:   500,
    QUEEN:  900,
    KING:   20000,
    EMPTY:  0,
}

# -----------------------------------------------------------------------
# Piece-Square Tables (from White's perspective, row 0 = rank 8)
# Values indicate positional bonuses in centipawns.
# -----------------------------------------------------------------------

PAWN_TABLE = np.array([
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
], dtype=np.int16)

KNIGHT_TABLE = np.array([
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
], dtype=np.int16)

BISHOP_TABLE = np.array([
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
], dtype=np.int16)

ROOK_TABLE = np.array([
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
], dtype=np.int16)

QUEEN_TABLE = np.array([
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
], dtype=np.int16)

KING_MIDDLEGAME_TABLE = np.array([
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
], dtype=np.int16)

KING_ENDGAME_TABLE = np.array([
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
], dtype=np.int16)

PIECE_TABLES = {
    PAWN:   PAWN_TABLE,
    KNIGHT: KNIGHT_TABLE,
    BISHOP: BISHOP_TABLE,
    ROOK:   ROOK_TABLE,
    QUEEN:  QUEEN_TABLE,
    KING:   KING_MIDDLEGAME_TABLE,
}


class Evaluator:
    """
    Static position evaluator.
    Returns score in centipawns from White's perspective.
    Positive = White advantage, Negative = Black advantage.
    """

    # Thresholds for game phase detection
    ENDGAME_MATERIAL = 1300  # Below this total, use endgame king table

    def evaluate(self, board: Board) -> int:
        """Full static evaluation of the board position."""
        score = 0
        total_material = 0

        for r in range(8):
            for c in range(8):
                piece = int(board.squares[r][c])
                if piece == EMPTY:
                    continue

                abs_piece = abs(piece)
                color = WHITE if piece > 0 else BLACK
                value = PIECE_VALUES.get(abs_piece, 0)

                if abs_piece != KING:
                    total_material += value

                # Material value (positive for White)
                score += value * color

                # Piece-square table bonus
                pst_score = self._get_pst_score(abs_piece, r, c, color, total_material)
                score += pst_score * color

        # Mobility bonus
        score += self._mobility_score(board)

        # King safety
        score += self._king_safety(board, WHITE, total_material)
        score -= self._king_safety(board, BLACK, total_material)

        # Pawn structure
        score += self._pawn_structure(board, WHITE)
        score -= self._pawn_structure(board, BLACK)

        # Bonus for bishop pair
        score += self._bishop_pair_bonus(board, WHITE)
        score -= self._bishop_pair_bonus(board, BLACK)

        return score

    def _get_pst_score(self, abs_piece: int, r: int, c: int,
                        color: int, total_material: int) -> int:
        """Get piece-square table bonus for a piece."""
        # Mirror rows for Black (Black's perspective is flipped)
        table_row = r if color == WHITE else (7 - r)
        idx = table_row * 8 + c

        if abs_piece == KING:
            if total_material < self.ENDGAME_MATERIAL:
                return int(KING_ENDGAME_TABLE[idx])
            else:
                return int(KING_MIDDLEGAME_TABLE[idx])

        table = PIECE_TABLES.get(abs_piece)
        if table is not None:
            return int(table[idx])
        return 0

    def _mobility_score(self, board: Board) -> int:
        """
        Score based on number of available moves (mobility).
        More moves = better position.
        """
        from move_generator import MoveGenerator
        mg = MoveGenerator(board)

        # Count pseudo-legal moves as a mobility proxy (faster than legal moves)
        white_moves = len(mg._generate_pseudo_legal_moves(WHITE))
        black_moves = len(mg._generate_pseudo_legal_moves(BLACK))

        return (white_moves - black_moves) * 5  # 5 cp per extra move

    def _king_safety(self, board: Board, color: int, total_material: int) -> int:
        """
        Evaluate king safety by checking pawn shield and open files near king.
        Only relevant in middlegame.
        """
        if total_material < self.ENDGAME_MATERIAL:
            return 0  # In endgame, centralize king instead

        score = 0
        king_pos = board.king_positions.get(color)
        if king_pos is None:
            return 0

        kr, kc = king_pos
        pawn = color * PAWN

        # Check pawn shield in front of king
        shield_row = kr - 1 if color == WHITE else kr + 1
        if 0 <= shield_row < 8:
            for dc in range(max(0, kc - 1), min(8, kc + 2)):
                if board.squares[shield_row][dc] == pawn:
                    score += 10  # Pawn shield bonus

        # Penalty for open files near king
        for dc in range(max(0, kc - 1), min(8, kc + 2)):
            file_open = True
            for row in range(8):
                if board.squares[row][dc] == pawn:
                    file_open = False
                    break
            if file_open:
                score -= 20  # Penalty for open file near king

        return score

    def _pawn_structure(self, board: Board, color: int) -> int:
        """Evaluate pawn structure: doubled, isolated, and passed pawns."""
        score = 0
        pawn = color * PAWN
        pawn_cols = []

        for r in range(8):
            for c in range(8):
                if board.squares[r][c] == pawn:
                    pawn_cols.append(c)

        for c in pawn_cols:
            # Doubled pawns penalty
            if pawn_cols.count(c) > 1:
                score -= 20

            # Isolated pawn penalty
            if (c - 1) not in pawn_cols and (c + 1) not in pawn_cols:
                score -= 15

        # Passed pawn bonus
        opp_pawn = -pawn
        for r in range(8):
            for c in range(8):
                if board.squares[r][c] == pawn:
                    if self._is_passed_pawn(board, r, c, color, opp_pawn):
                        # Bonus increases as pawn advances
                        advance = (6 - r) if color == WHITE else (r - 1)
                        score += 20 + advance * 15

        return score

    def _is_passed_pawn(self, board: Board, r: int, c: int,
                          color: int, opp_pawn: int) -> bool:
        """Check if a pawn has no opposing pawns blocking or adjacent."""
        direction = -1 if color == WHITE else 1
        check_row = r + direction
        while 0 <= check_row < 8:
            for dc in range(max(0, c - 1), min(8, c + 2)):
                if board.squares[check_row][dc] == opp_pawn:
                    return False
            check_row += direction
        return True

    def _bishop_pair_bonus(self, board: Board, color: int) -> int:
        """Grant bonus for having both bishops."""
        bishop = color * BISHOP
        count = sum(
            1 for r in range(8) for c in range(8)
            if board.squares[r][c] == bishop
        )
        return 30 if count >= 2 else 0

    def evaluate_for_side(self, board: Board, color: int) -> int:
        """Return evaluation from the perspective of the given color."""
        score = self.evaluate(board)
        return score if color == WHITE else -score
