"""
opening_book.py - Chess Opening Book

Loads pre-computed opening moves from a pickle file and provides
move selection during the opening phase of the game.

Falls back to engine search if position not found in book,
or if the book file doesn't exist.
"""

import os
import pickle
import random
from typing import Optional, Dict, List, Tuple
from board import Board, Move, WHITE, BLACK, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING


# Common opening moves in algebraic notation for book generation
OPENING_LINES = [
    # e4 openings
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"],        # Ruy Lopez
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"],        # Italian Game
    ["e2e4", "e7e5", "g1f3", "b8c6", "d2d4"],        # Scotch Game
    ["e2e4", "c7c5"],                                   # Sicilian Defense
    ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4"],        # Sicilian Open
    ["e2e4", "e7e6"],                                   # French Defense
    ["e2e4", "e7e6", "d2d4", "d7d5"],                # French main line
    ["e2e4", "c7c6"],                                   # Caro-Kann
    ["e2e4", "d7d5"],                                   # Scandinavian
    # d4 openings
    ["d2d4", "d7d5", "c2c4"],                          # Queen's Gambit
    ["d2d4", "d7d5", "c2c4", "e7e6"],                # Queen's Gambit Declined
    ["d2d4", "d7d5", "c2c4", "c7c6"],                # Slav Defense
    ["d2d4", "g8f6", "c2c4"],                          # Indian Defenses
    ["d2d4", "g8f6", "c2c4", "g7g6"],                # King's Indian Defense
    ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "f8b4"],  # Nimzo-Indian
    # Flank openings
    ["c2c4"],                                            # English Opening
    ["g1f3"],                                            # Reti Opening
    ["g1f3", "d7d5", "c2c4"],                          # Reti main line
]


def generate_opening_book() -> Dict[str, List[str]]:
    """
    Generate a simple opening book from predefined opening lines.
    The book maps FEN prefixes to lists of candidate moves.
    """
    book: Dict[str, List[str]] = {}

    for line in OPENING_LINES:
        board = Board()
        for move_str in line:
            fen = board.get_fen().split(' ')[0]  # Position part of FEN
            turn = 'w' if board.turn == WHITE else 'b'
            key = f"{fen} {turn}"

            if key not in book:
                book[key] = []
            if move_str not in book[key]:
                book[key].append(move_str)

            # Execute the move
            move = _parse_algebraic(board, move_str)
            if move:
                board.make_move(move)
            else:
                break

    return book


def _parse_algebraic(board: Board, move_str: str) -> Optional[Move]:
    """
    Parse a move in long algebraic notation (e.g., 'e2e4', 'e7e8q').
    Returns a Move object or None if invalid.
    """
    if len(move_str) < 4:
        return None

    files = 'abcdefgh'
    try:
        fc = files.index(move_str[0])
        fr = 8 - int(move_str[1])
        tc = files.index(move_str[2])
        tr = 8 - int(move_str[3])
    except (ValueError, IndexError):
        return None

    # Promotion piece
    promotion = 0
    if len(move_str) == 5:
        promo_map = {'q': QUEEN, 'r': ROOK, 'b': BISHOP, 'n': KNIGHT}
        promo_piece = promo_map.get(move_str[4].lower(), 0)
        if promo_piece:
            promotion = promo_piece * board.turn

    from move_generator import MoveGenerator
    legal_moves = MoveGenerator(board).generate_legal_moves(board.turn)

    for move in legal_moves:
        if (move.from_sq == (fr, fc) and move.to_sq == (tr, tc)):
            if promotion == 0 or move.promotion == promotion:
                return move

    return None


class OpeningBook:
    """
    Manages loading and querying the opening book.

    The book is stored as a pickle file mapping position keys to
    lists of candidate moves in algebraic notation.
    """

    def __init__(self, book_path: str = "data/opening_book.pkl"):
        self.book_path = book_path
        self.book: Dict[str, List[str]] = {}
        self.enabled = True
        self._load_or_generate()

    def _load_or_generate(self):
        """Load book from file, or generate and save it if missing."""
        if os.path.exists(self.book_path):
            try:
                with open(self.book_path, 'rb') as f:
                    self.book = pickle.load(f)
                print(f"[OpeningBook] Loaded {len(self.book)} positions from {self.book_path}")
                return
            except Exception as e:
                print(f"[OpeningBook] Error loading book: {e}")

        # Generate and save
        print("[OpeningBook] Generating opening book...")
        self.book = generate_opening_book()
        self._save()
        print(f"[OpeningBook] Generated {len(self.book)} positions")

    def _save(self):
        """Save the opening book to disk."""
        os.makedirs(os.path.dirname(self.book_path) if os.path.dirname(self.book_path) else '.', exist_ok=True)
        try:
            with open(self.book_path, 'wb') as f:
                pickle.dump(self.book, f)
        except Exception as e:
            print(f"[OpeningBook] Error saving: {e}")

    def get_move(self, board: Board) -> Optional[Move]:
        """
        Look up a move for the current position.
        Returns a randomly selected book move, or None if not in book.
        """
        if not self.enabled:
            return None

        # Only use book in the first 15 moves
        if board.fullmove_number > 15:
            return None

        fen = board.get_fen().split(' ')[0]
        turn = 'w' if board.turn == WHITE else 'b'
        key = f"{fen} {turn}"

        candidates = self.book.get(key, [])
        if not candidates:
            return None

        # Select a random candidate move
        random.shuffle(candidates)
        for move_str in candidates:
            move = _parse_algebraic(board, move_str)
            if move:
                print(f"[OpeningBook] Book move: {move_str}")
                return move

        return None

    def has_position(self, board: Board) -> bool:
        """Check if current position is in the opening book."""
        fen = board.get_fen().split(' ')[0]
        turn = 'w' if board.turn == WHITE else 'b'
        key = f"{fen} {turn}"
        return key in self.book

    def disable(self):
        """Disable the opening book (for testing/analysis)."""
        self.enabled = False

    def enable(self):
        """Re-enable the opening book."""
        self.enabled = True
