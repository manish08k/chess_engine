"""
board.py - Chess Board Representation and Game State Management

This module implements the core chess board using numpy arrays for
efficient piece representation, move execution, undo functionality,
and rule enforcement including castling, en passant, and promotion.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict

# Piece constants (positive = White, negative = Black)
EMPTY  = 0
PAWN   = 1
KNIGHT = 2
BISHOP = 3
ROOK   = 4
QUEEN  = 5
KING   = 6

WHITE = 1
BLACK = -1

PIECE_NAMES = {
    EMPTY: '.',
    PAWN: 'P', KNIGHT: 'N', BISHOP: 'B', ROOK: 'R', QUEEN: 'Q', KING: 'K',
    -PAWN: 'p', -KNIGHT: 'n', -BISHOP: 'b', -ROOK: 'r', -QUEEN: 'q', -KING: 'k',
}

# Castling rights bitmask
CASTLE_WK = 0b0001  # White kingside
CASTLE_WQ = 0b0010  # White queenside
CASTLE_BK = 0b0100  # Black kingside
CASTLE_BQ = 0b1000  # Black queenside


class Move:
    """Represents a chess move with all metadata needed for undo."""

    __slots__ = [
        'from_sq', 'to_sq', 'piece', 'captured',
        'promotion', 'is_castling', 'is_en_passant',
        'prev_ep_square', 'prev_castling_rights', 'prev_halfmove'
    ]

    def __init__(
        self,
        from_sq: Tuple[int, int],
        to_sq: Tuple[int, int],
        piece: int,
        captured: int = EMPTY,
        promotion: int = EMPTY,
        is_castling: bool = False,
        is_en_passant: bool = False,
    ):
        self.from_sq = from_sq
        self.to_sq = to_sq
        self.piece = piece
        self.captured = captured
        self.promotion = promotion
        self.is_castling = is_castling
        self.is_en_passant = is_en_passant
        # Saved state for undo
        self.prev_ep_square: Optional[Tuple[int, int]] = None
        self.prev_castling_rights: int = 0
        self.prev_halfmove: int = 0

    def __repr__(self) -> str:
        files = 'abcdefgh'
        fr, fc = self.from_sq
        tr, tc = self.to_sq
        promo = f'={PIECE_NAMES.get(abs(self.promotion), "")}' if self.promotion else ''
        return f"{files[fc]}{8 - fr}{files[tc]}{8 - tr}{promo}"

    def __eq__(self, other) -> bool:
        return (self.from_sq == other.from_sq and
                self.to_sq == other.to_sq and
                self.promotion == other.promotion)

    def __hash__(self) -> int:
        return hash((self.from_sq, self.to_sq, self.promotion))


class Board:
    """
    Chess board using an 8x8 numpy int8 array.

    Positive values = White pieces, negative = Black pieces.
    Tracks full game state: turn, castling rights, en passant, move counters.
    """

    def __init__(self):
        self.squares: np.ndarray = np.zeros((8, 8), dtype=np.int8)
        self.turn: int = WHITE
        self.castling_rights: int = CASTLE_WK | CASTLE_WQ | CASTLE_BK | CASTLE_BQ
        self.en_passant_square: Optional[Tuple[int, int]] = None
        self.halfmove_clock: int = 0   # For 50-move rule
        self.fullmove_number: int = 1
        self.move_history: List[Move] = []
        self.king_positions: Dict[int, Tuple[int, int]] = {WHITE: (7, 4), BLACK: (0, 4)}
        self._setup_starting_position()

    # ------------------------------------------------------------------
    # Board Setup
    # ------------------------------------------------------------------

    def _setup_starting_position(self):
        """Initialize board to standard chess starting position."""
        back_rank = [ROOK, KNIGHT, BISHOP, QUEEN, KING, BISHOP, KNIGHT, ROOK]

        # Black back rank (row 0)
        for col, piece in enumerate(back_rank):
            self.squares[0][col] = -piece
        # Black pawns (row 1)
        self.squares[1] = np.full(8, -PAWN, dtype=np.int8)

        # White pawns (row 6)
        self.squares[6] = np.full(8, PAWN, dtype=np.int8)
        # White back rank (row 7)
        for col, piece in enumerate(back_rank):
            self.squares[7][col] = piece

        self.king_positions = {WHITE: (7, 4), BLACK: (0, 4)}

    def copy(self) -> 'Board':
        """Return a deep copy of the board state."""
        new_board = Board.__new__(Board)
        new_board.squares = self.squares.copy()
        new_board.turn = self.turn
        new_board.castling_rights = self.castling_rights
        new_board.en_passant_square = self.en_passant_square
        new_board.halfmove_clock = self.halfmove_clock
        new_board.fullmove_number = self.fullmove_number
        new_board.move_history = list(self.move_history)
        new_board.king_positions = dict(self.king_positions)
        return new_board

    # ------------------------------------------------------------------
    # Move Execution
    # ------------------------------------------------------------------

    def make_move(self, move: Move) -> bool:
        """
        Execute a move on the board. Returns False if the move leaves
        own king in check (illegal move).
        """
        # Save state for undo
        move.prev_ep_square = self.en_passant_square
        move.prev_castling_rights = self.castling_rights
        move.prev_halfmove = self.halfmove_clock

        fr, fc = move.from_sq
        tr, tc = move.to_sq
        piece = self.squares[fr][fc]
        color = WHITE if piece > 0 else BLACK

        # Update halfmove clock
        if abs(piece) == PAWN or move.captured != EMPTY:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        # Clear en passant square
        self.en_passant_square = None

        # --- Execute piece movement ---
        self.squares[tr][tc] = piece
        self.squares[fr][fc] = EMPTY

        # Track king position
        if abs(piece) == KING:
            self.king_positions[color] = (tr, tc)

        # Handle special moves
        if move.is_en_passant:
            # Remove the captured pawn
            ep_row = fr  # Captured pawn is on same row as moving pawn
            self.squares[ep_row][tc] = EMPTY

        elif move.is_castling:
            # Move the rook
            if tc == 6:  # Kingside
                self.squares[tr][5] = self.squares[tr][7]
                self.squares[tr][7] = EMPTY
            else:  # Queenside
                self.squares[tr][3] = self.squares[tr][0]
                self.squares[tr][0] = EMPTY

        elif move.promotion != EMPTY:
            self.squares[tr][tc] = move.promotion

        # Set en passant square for double pawn push
        if abs(piece) == PAWN and abs(tr - fr) == 2:
            ep_row = (fr + tr) // 2
            self.en_passant_square = (ep_row, fc)

        # Update castling rights
        self._update_castling_rights(fr, fc, tr, tc)

        # Update move counters
        if color == BLACK:
            self.fullmove_number += 1
        self.turn = -self.turn

        # Verify move doesn't leave own king in check
        if self.is_in_check(color):
            self.undo_move(move)
            return False

        self.move_history.append(move)
        return True

    def undo_move(self, move: Move):
        """Restore board to state before the given move."""
        fr, fc = move.from_sq
        tr, tc = move.to_sq

        # Restore moved piece
        piece = self.squares[tr][tc]
        if move.promotion != EMPTY:
            # Restore pawn (use sign to get color)
            piece = PAWN * (WHITE if move.promotion > 0 else BLACK)

        self.squares[fr][fc] = piece
        self.squares[tr][tc] = move.captured

        color = WHITE if piece > 0 else BLACK

        # Track king position
        if abs(piece) == KING:
            self.king_positions[color] = (fr, fc)

        # Undo en passant capture
        if move.is_en_passant:
            ep_pawn = -PAWN * color  # Opponent's pawn
            self.squares[fr][tc] = ep_pawn

        # Undo castling rook
        elif move.is_castling:
            if tc == 6:  # Kingside
                self.squares[tr][7] = self.squares[tr][5]
                self.squares[tr][5] = EMPTY
            else:  # Queenside
                self.squares[tr][0] = self.squares[tr][3]
                self.squares[tr][3] = EMPTY

        # Restore saved state
        self.en_passant_square = move.prev_ep_square
        self.castling_rights = move.prev_castling_rights
        self.halfmove_clock = move.prev_halfmove
        self.turn = -self.turn

        if self.turn == BLACK:
            self.fullmove_number -= 1

        if self.move_history and self.move_history[-1] == move:
            self.move_history.pop()

    def _update_castling_rights(self, fr: int, fc: int, tr: int, tc: int):
        """Remove castling rights when king or rook moves."""
        # King moves
        if (fr, fc) == (7, 4):
            self.castling_rights &= ~(CASTLE_WK | CASTLE_WQ)
        elif (fr, fc) == (0, 4):
            self.castling_rights &= ~(CASTLE_BK | CASTLE_BQ)

        # Rook moves or is captured
        if (fr, fc) == (7, 7) or (tr, tc) == (7, 7):
            self.castling_rights &= ~CASTLE_WK
        if (fr, fc) == (7, 0) or (tr, tc) == (7, 0):
            self.castling_rights &= ~CASTLE_WQ
        if (fr, fc) == (0, 7) or (tr, tc) == (0, 7):
            self.castling_rights &= ~CASTLE_BK
        if (fr, fc) == (0, 0) or (tr, tc) == (0, 0):
            self.castling_rights &= ~CASTLE_BQ

    # ------------------------------------------------------------------
    # Check / Checkmate Detection
    # ------------------------------------------------------------------

    def is_in_check(self, color: int) -> bool:
        """Return True if the given color's king is under attack."""
        king_pos = self.king_positions.get(color)
        if king_pos is None:
            return False
        return self.is_square_attacked(king_pos, -color)

    def is_square_attacked(self, square: Tuple[int, int], by_color: int) -> bool:
        """Return True if the given square is attacked by any piece of by_color."""
        r, c = square

        # Check knight attacks
        for dr, dc in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                if self.squares[nr][nc] == by_color * KNIGHT:
                    return True

        # Check pawn attacks
        pawn_dir = -by_color  # Direction pawns move (relative to by_color)
        for dc in [-1, 1]:
            nr, nc = r + pawn_dir, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                if self.squares[nr][nc] == by_color * PAWN:
                    return True

        # Check king attacks
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < 8 and 0 <= nc < 8:
                    if self.squares[nr][nc] == by_color * KING:
                        return True

        # Check sliding pieces (rook/queen on ranks/files)
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r + dr, c + dc
            while 0 <= nr < 8 and 0 <= nc < 8:
                piece = self.squares[nr][nc]
                if piece != EMPTY:
                    if piece == by_color * ROOK or piece == by_color * QUEEN:
                        return True
                    break
                nr += dr
                nc += dc

        # Check sliding pieces (bishop/queen on diagonals)
        for dr, dc in [(1,1),(1,-1),(-1,1),(-1,-1)]:
            nr, nc = r + dr, c + dc
            while 0 <= nr < 8 and 0 <= nc < 8:
                piece = self.squares[nr][nc]
                if piece != EMPTY:
                    if piece == by_color * BISHOP or piece == by_color * QUEEN:
                        return True
                    break
                nr += dr
                nc += dc

        return False

    def is_checkmate(self, color: int) -> bool:
        """Return True if color has no legal moves and is in check."""
        from move_generator import MoveGenerator
        mg = MoveGenerator(self)
        return self.is_in_check(color) and len(mg.generate_legal_moves(color)) == 0

    def is_stalemate(self, color: int) -> bool:
        """Return True if color has no legal moves but is not in check."""
        from move_generator import MoveGenerator
        mg = MoveGenerator(self)
        return not self.is_in_check(color) and len(mg.generate_legal_moves(color)) == 0

    def is_draw_by_fifty_moves(self) -> bool:
        return self.halfmove_clock >= 100

    def is_insufficient_material(self) -> bool:
        """Check for draws by insufficient material."""
        pieces = {WHITE: [], BLACK: []}
        for r in range(8):
            for c in range(8):
                p = self.squares[r][c]
                if p != EMPTY:
                    color = WHITE if p > 0 else BLACK
                    pieces[color].append(abs(p))

        for color in [WHITE, BLACK]:
            p = sorted(pieces[color])
            if p == [KING]:
                opp = pieces[-color]
                if sorted(opp) in [[KING], [KING, BISHOP], [KING, KNIGHT]]:
                    return True
        return False

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_piece_at(self, square: Tuple[int, int]) -> int:
        r, c = square
        return int(self.squares[r][c])

    def display(self) -> str:
        """Return ASCII representation of the board."""
        rows = []
        rows.append("  a b c d e f g h")
        for r in range(8):
            row_str = f"{8 - r} "
            for c in range(8):
                row_str += PIECE_NAMES.get(int(self.squares[r][c]), '?') + ' '
            rows.append(row_str)
        rows.append("  a b c d e f g h")
        return '\n'.join(rows)

    def get_fen(self) -> str:
        """Generate FEN string for current position."""
        fen_parts = []
        for r in range(8):
            empty = 0
            row_str = ''
            for c in range(8):
                p = int(self.squares[r][c])
                if p == EMPTY:
                    empty += 1
                else:
                    if empty:
                        row_str += str(empty)
                        empty = 0
                    row_str += PIECE_NAMES.get(p, '?')
            if empty:
                row_str += str(empty)
            fen_parts.append(row_str)

        position = '/'.join(fen_parts)
        turn_str = 'w' if self.turn == WHITE else 'b'

        castle_str = ''
        if self.castling_rights & CASTLE_WK: castle_str += 'K'
        if self.castling_rights & CASTLE_WQ: castle_str += 'Q'
        if self.castling_rights & CASTLE_BK: castle_str += 'k'
        if self.castling_rights & CASTLE_BQ: castle_str += 'q'
        if not castle_str: castle_str = '-'

        ep_str = '-'
        if self.en_passant_square:
            er, ec = self.en_passant_square
            ep_str = 'abcdefgh'[ec] + str(8 - er)

        return f"{position} {turn_str} {castle_str} {ep_str} {self.halfmove_clock} {self.fullmove_number}"
