"""
move_generator.py - Legal Chess Move Generation

Generates all legal moves for a given board position.
Handles sliding pieces, special moves (castling, en passant, promotion),
and filters moves that leave the king in check.
"""

from typing import List, Tuple, Optional
from board import (
    Board, Move, WHITE, BLACK,
    EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
    CASTLE_WK, CASTLE_WQ, CASTLE_BK, CASTLE_BQ
)


class MoveGenerator:
    """Generates pseudo-legal and legal moves for a chess position."""

    def __init__(self, board: Board):
        self.board = board

    def generate_legal_moves(self, color: int) -> List[Move]:
        """Generate all legal moves for the given color."""
        pseudo = self._generate_pseudo_legal_moves(color)
        legal = []
        for move in pseudo:
            if self.board.make_move(move):
                # make_move returns True only if legal (king not in check)
                legal.append(move)
                self.board.undo_move(move)
        return legal

    def _generate_pseudo_legal_moves(self, color: int) -> List[Move]:
        """Generate all pseudo-legal moves (may leave king in check)."""
        moves: List[Move] = []
        board = self.board

        for r in range(8):
            for c in range(8):
                piece = int(board.squares[r][c])
                if piece == EMPTY:
                    continue
                piece_color = WHITE if piece > 0 else BLACK
                if piece_color != color:
                    continue

                abs_piece = abs(piece)
                if abs_piece == PAWN:
                    moves.extend(self._pawn_moves(r, c, color))
                elif abs_piece == KNIGHT:
                    moves.extend(self._knight_moves(r, c, color))
                elif abs_piece == BISHOP:
                    moves.extend(self._sliding_moves(r, c, color, [(1,1),(1,-1),(-1,1),(-1,-1)]))
                elif abs_piece == ROOK:
                    moves.extend(self._sliding_moves(r, c, color, [(0,1),(0,-1),(1,0),(-1,0)]))
                elif abs_piece == QUEEN:
                    moves.extend(self._sliding_moves(r, c, color,
                        [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]))
                elif abs_piece == KING:
                    moves.extend(self._king_moves(r, c, color))

        moves.extend(self._castling_moves(color))
        return moves

    # ------------------------------------------------------------------
    # Pawn Moves
    # ------------------------------------------------------------------

    def _pawn_moves(self, r: int, c: int, color: int) -> List[Move]:
        moves: List[Move] = []
        board = self.board
        direction = -1 if color == WHITE else 1  # White moves up (decreasing row)
        start_row = 6 if color == WHITE else 1
        promo_row = 0 if color == WHITE else 7

        # Single push
        nr = r + direction
        if 0 <= nr < 8 and board.squares[nr][c] == EMPTY:
            if nr == promo_row:
                for promo in [QUEEN, ROOK, BISHOP, KNIGHT]:
                    moves.append(Move((r, c), (nr, c), color * PAWN,
                                      promotion=color * promo))
            else:
                moves.append(Move((r, c), (nr, c), color * PAWN))

            # Double push from starting rank
            if r == start_row:
                nr2 = r + 2 * direction
                if board.squares[nr2][c] == EMPTY:
                    moves.append(Move((r, c), (nr2, c), color * PAWN))

        # Captures
        for dc in [-1, 1]:
            nc = c + dc
            if not (0 <= nc < 8):
                continue
            nr = r + direction

            # Normal capture
            target = int(board.squares[nr][nc])
            if target != EMPTY and (WHITE if target > 0 else BLACK) != color:
                if nr == promo_row:
                    for promo in [QUEEN, ROOK, BISHOP, KNIGHT]:
                        moves.append(Move((r, c), (nr, nc), color * PAWN,
                                          captured=target, promotion=color * promo))
                else:
                    moves.append(Move((r, c), (nr, nc), color * PAWN, captured=target))

            # En passant capture
            elif board.en_passant_square == (nr, nc):
                captured_pawn = -color * PAWN
                moves.append(Move((r, c), (nr, nc), color * PAWN,
                                  captured=captured_pawn, is_en_passant=True))

        return moves

    # ------------------------------------------------------------------
    # Knight Moves
    # ------------------------------------------------------------------

    def _knight_moves(self, r: int, c: int, color: int) -> List[Move]:
        moves: List[Move] = []
        board = self.board
        piece = int(board.squares[r][c])

        for dr, dc in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < 8 and 0 <= nc < 8):
                continue
            target = int(board.squares[nr][nc])
            if target == EMPTY:
                moves.append(Move((r, c), (nr, nc), piece))
            elif (WHITE if target > 0 else BLACK) != color:
                moves.append(Move((r, c), (nr, nc), piece, captured=target))

        return moves

    # ------------------------------------------------------------------
    # Sliding Piece Moves (Bishop, Rook, Queen)
    # ------------------------------------------------------------------

    def _sliding_moves(self, r: int, c: int, color: int,
                        directions: List[Tuple[int, int]]) -> List[Move]:
        moves: List[Move] = []
        board = self.board
        piece = int(board.squares[r][c])

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            while 0 <= nr < 8 and 0 <= nc < 8:
                target = int(board.squares[nr][nc])
                if target == EMPTY:
                    moves.append(Move((r, c), (nr, nc), piece))
                elif (WHITE if target > 0 else BLACK) != color:
                    # Capture and stop
                    moves.append(Move((r, c), (nr, nc), piece, captured=target))
                    break
                else:
                    # Blocked by own piece
                    break
                nr += dr
                nc += dc

        return moves

    # ------------------------------------------------------------------
    # King Moves
    # ------------------------------------------------------------------

    def _king_moves(self, r: int, c: int, color: int) -> List[Move]:
        moves: List[Move] = []
        board = self.board
        piece = int(board.squares[r][c])

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if not (0 <= nr < 8 and 0 <= nc < 8):
                    continue
                target = int(board.squares[nr][nc])
                if target == EMPTY:
                    moves.append(Move((r, c), (nr, nc), piece))
                elif (WHITE if target > 0 else BLACK) != color:
                    moves.append(Move((r, c), (nr, nc), piece, captured=target))

        return moves

    # ------------------------------------------------------------------
    # Castling Moves
    # ------------------------------------------------------------------

    def _castling_moves(self, color: int) -> List[Move]:
        moves: List[Move] = []
        board = self.board

        if color == WHITE:
            king_row = 7
            king_col = 4
            # Check king is in expected position
            if board.squares[king_row][king_col] != KING:
                return moves
            if board.is_in_check(WHITE):
                return moves

            # Kingside castling
            if (board.castling_rights & CASTLE_WK and
                    board.squares[7][5] == EMPTY and
                    board.squares[7][6] == EMPTY and
                    board.squares[7][7] == ROOK):
                if (not board.is_square_attacked((7, 5), BLACK) and
                        not board.is_square_attacked((7, 6), BLACK)):
                    moves.append(Move((7, 4), (7, 6), KING, is_castling=True))

            # Queenside castling
            if (board.castling_rights & CASTLE_WQ and
                    board.squares[7][3] == EMPTY and
                    board.squares[7][2] == EMPTY and
                    board.squares[7][1] == EMPTY and
                    board.squares[7][0] == ROOK):
                if (not board.is_square_attacked((7, 3), BLACK) and
                        not board.is_square_attacked((7, 2), BLACK)):
                    moves.append(Move((7, 4), (7, 2), KING, is_castling=True))

        else:  # BLACK
            king_row = 0
            king_col = 4
            if board.squares[king_row][king_col] != -KING:
                return moves
            if board.is_in_check(BLACK):
                return moves

            # Kingside castling
            if (board.castling_rights & CASTLE_BK and
                    board.squares[0][5] == EMPTY and
                    board.squares[0][6] == EMPTY and
                    board.squares[0][7] == -ROOK):
                if (not board.is_square_attacked((0, 5), WHITE) and
                        not board.is_square_attacked((0, 6), WHITE)):
                    moves.append(Move((0, 4), (0, 6), -KING, is_castling=True))

            # Queenside castling
            if (board.castling_rights & CASTLE_BQ and
                    board.squares[0][3] == EMPTY and
                    board.squares[0][2] == EMPTY and
                    board.squares[0][1] == EMPTY and
                    board.squares[0][0] == -ROOK):
                if (not board.is_square_attacked((0, 3), WHITE) and
                        not board.is_square_attacked((0, 2), WHITE)):
                    moves.append(Move((0, 4), (0, 2), -KING, is_castling=True))

        return moves

    # ------------------------------------------------------------------
    # Move Ordering (for alpha-beta efficiency)
    # ------------------------------------------------------------------

    def order_moves(self, moves: List[Move], tt_move: Optional[Move] = None) -> List[Move]:
        """
        Order moves for better alpha-beta pruning performance.
        Priority: TT move > captures (MVV-LVA) > quiet moves.
        """
        PIECE_VALUES = {PAWN: 100, KNIGHT: 320, BISHOP: 330,
                        ROOK: 500, QUEEN: 900, KING: 20000, EMPTY: 0}

        def score(move: Move) -> int:
            if tt_move and move == tt_move:
                return 100000  # Highest priority
            if move.captured != EMPTY:
                # MVV-LVA: Most Valuable Victim - Least Valuable Attacker
                victim = PIECE_VALUES.get(abs(move.captured), 0)
                attacker = PIECE_VALUES.get(abs(move.piece), 0)
                return 10000 + victim * 10 - attacker
            if move.promotion != EMPTY:
                return 9000
            return 0

        return sorted(moves, key=score, reverse=True)
