"""
zobrist_hash.py - Zobrist Hashing for Transposition Tables

Zobrist hashing assigns random 64-bit integers to each (piece, square)
combination, enabling fast incremental board hashing. Used to detect
repeated positions and cache search results in the transposition table.
"""

import numpy as np
from typing import Optional, Dict, Tuple, Any
from board import WHITE, BLACK, EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING


# Piece indices for the Zobrist table
# White pieces: 0-5, Black pieces: 6-11
PIECE_INDEX = {
    PAWN: 0, KNIGHT: 1, BISHOP: 2, ROOK: 3, QUEEN: 4, KING: 5,
    -PAWN: 6, -KNIGHT: 7, -BISHOP: 8, -ROOK: 9, -QUEEN: 10, -KING: 11,
}

NUM_PIECE_TYPES = 12
NUM_SQUARES = 64


class ZobristHash:
    """
    Implements Zobrist hashing for chess positions.

    The hash is computed by XOR-ing random 64-bit integers for:
    - Each piece on each square
    - Side to move (if Black)
    - Castling rights (4 bits)
    - En passant file (if applicable)
    """

    def __init__(self, seed: int = 42):
        rng = np.random.default_rng(seed)
        # Table[piece_type][square] -> random 64-bit integer
        self.piece_table: np.ndarray = rng.integers(
            low=0, high=np.iinfo(np.uint64).max, dtype=np.uint64,
            size=(NUM_PIECE_TYPES, NUM_SQUARES)
        )
        # XOR when it's Black's turn
        self.black_to_move: np.uint64 = rng.integers(
            0, np.iinfo(np.uint64).max, dtype=np.uint64
        )
        # XOR for each castling right bit
        self.castling_table: np.ndarray = rng.integers(
            low=0, high=np.iinfo(np.uint64).max, dtype=np.uint64, size=16
        )
        # XOR for en passant file (0-7)
        self.ep_table: np.ndarray = rng.integers(
            low=0, high=np.iinfo(np.uint64).max, dtype=np.uint64, size=8
        )

    def compute_hash(self, board) -> np.uint64:
        """Compute full Zobrist hash from scratch for a board position."""
        h = np.uint64(0)

        for r in range(8):
            for c in range(8):
                piece = int(board.squares[r][c])
                if piece == EMPTY:
                    continue
                piece_idx = PIECE_INDEX.get(piece)
                if piece_idx is None:
                    continue
                sq = r * 8 + c
                h ^= self.piece_table[piece_idx][sq]

        if board.turn == BLACK:
            h ^= self.black_to_move

        h ^= self.castling_table[board.castling_rights & 0xF]

        if board.en_passant_square is not None:
            _, ep_col = board.en_passant_square
            h ^= self.ep_table[ep_col]

        return h

    def update_hash(self, h: np.uint64, move, prev_castling: int,
                    prev_ep_col: Optional[int]) -> np.uint64:
        """
        Incrementally update hash after making a move.
        Much faster than recomputing from scratch.
        """
        from board import Move

        fr, fc = move.from_sq
        tr, tc = move.to_sq
        piece = move.piece

        from_sq = fr * 8 + fc
        to_sq = tr * 8 + tc

        # Remove piece from source square
        piece_idx = PIECE_INDEX.get(piece)
        if piece_idx is not None:
            h ^= self.piece_table[piece_idx][from_sq]

        # Remove captured piece
        if move.captured != EMPTY:
            if move.is_en_passant:
                # Captured pawn is on the source row, destination column
                cap_sq = fr * 8 + tc
            else:
                cap_sq = to_sq
            cap_idx = PIECE_INDEX.get(move.captured)
            if cap_idx is not None:
                h ^= self.piece_table[cap_idx][cap_sq]

        # Add piece to destination square (handle promotion)
        dest_piece = move.promotion if move.promotion != 0 else piece
        dest_idx = PIECE_INDEX.get(dest_piece)
        if dest_idx is not None:
            h ^= self.piece_table[dest_idx][to_sq]

        # Handle castling rook
        if move.is_castling:
            color = 1 if piece > 0 else -1
            if tc == 6:  # Kingside
                rook_from = (fr * 8 + 7)
                rook_to = (tr * 8 + 5)
            else:  # Queenside
                rook_from = (fr * 8 + 0)
                rook_to = (tr * 8 + 3)
            rook_piece = 4 * color  # ROOK * color
            rook_idx = PIECE_INDEX.get(rook_piece)
            if rook_idx is not None:
                h ^= self.piece_table[rook_idx][rook_from]
                h ^= self.piece_table[rook_idx][rook_to]

        # Toggle side to move
        h ^= self.black_to_move

        # Update castling rights (XOR out old, XOR in new handled by board)
        # We pass prev_castling and let caller handle new rights
        h ^= self.castling_table[prev_castling & 0xF]
        # Note: caller adds new castling hash after computing new rights

        # Update en passant
        if prev_ep_col is not None:
            h ^= self.ep_table[prev_ep_col]

        return h


# -----------------------------------------------------------------------
# Transposition Table
# -----------------------------------------------------------------------

TT_EXACT = 0   # Exact score
TT_ALPHA = 1   # Upper bound (score <= alpha)
TT_BETA  = 2   # Lower bound (score >= beta)

class TTEntry:
    """Entry in the transposition table."""
    __slots__ = ['hash', 'depth', 'score', 'flag', 'best_move']

    def __init__(self, hash_val: np.uint64, depth: int, score: int,
                 flag: int, best_move=None):
        self.hash = hash_val
        self.depth = depth
        self.score = score
        self.flag = flag
        self.best_move = best_move


class TranspositionTable:
    """
    Fixed-size transposition table using Zobrist hashes.
    Uses a simple array with index = hash % size.
    Replacement strategy: replace if depth >= stored depth.
    """

    def __init__(self, size_mb: int = 32):
        # Approximate number of entries for given MB
        entry_size_bytes = 64  # Rough estimate per Python object
        self.size = (size_mb * 1024 * 1024) // entry_size_bytes
        self.table: Dict[int, TTEntry] = {}
        self.hits = 0
        self.stores = 0

    def lookup(self, hash_val: np.uint64, depth: int, alpha: int,
               beta: int) -> Tuple[Optional[int], Optional[Any]]:
        """
        Look up position in TT. Returns (score, best_move) or (None, None).
        """
        key = int(hash_val) % self.size
        entry = self.table.get(key)
        if entry is None or entry.hash != hash_val:
            return None, None

        if entry.depth >= depth:
            self.hits += 1
            if entry.flag == TT_EXACT:
                return entry.score, entry.best_move
            elif entry.flag == TT_ALPHA and entry.score <= alpha:
                return alpha, entry.best_move
            elif entry.flag == TT_BETA and entry.score >= beta:
                return beta, entry.best_move

        return None, entry.best_move  # Return best_move for move ordering

    def store(self, hash_val: np.uint64, depth: int, score: int,
              flag: int, best_move=None):
        """Store a position in the transposition table."""
        key = int(hash_val) % self.size
        existing = self.table.get(key)

        # Replace if new entry is deeper or slot is empty
        if existing is None or existing.depth <= depth:
            self.table[key] = TTEntry(hash_val, depth, score, flag, best_move)
            self.stores += 1

        # Limit table size to prevent memory bloat
        if len(self.table) > self.size:
            # Simple cleanup: remove ~10% of oldest entries
            keys = list(self.table.keys())[:self.size // 10]
            for k in keys:
                del self.table[k]

    def clear(self):
        """Clear the transposition table."""
        self.table.clear()
        self.hits = 0
        self.stores = 0
