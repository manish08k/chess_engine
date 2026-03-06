"""
minimax.py - Minimax Search Algorithm

Implements the classic minimax algorithm with depth-limited search.
The maximizing player (White) tries to maximize score, while the
minimizing player (Black) tries to minimize it.

This module also serves as a baseline for comparison with alpha-beta pruning.
"""

from typing import Optional, Tuple
from board import Board, Move, WHITE, BLACK
from evaluation import Evaluator
from move_generator import MoveGenerator

INF = 10_000_000  # Effectively infinite score


class MinimaxSearch:
    """
    Pure minimax search without pruning.
    Useful for testing/debugging; for production use AlphaBetaSearch.
    """

    def __init__(self, board: Board, max_depth: int = 3):
        self.board = board
        self.max_depth = max_depth
        self.evaluator = Evaluator()
        self.nodes_searched = 0
        self.best_move: Optional[Move] = None

    def search(self) -> Optional[Move]:
        """
        Find the best move from the current position.
        Returns the best Move found, or None if no legal moves.
        """
        self.nodes_searched = 0
        self.best_move = None
        color = self.board.turn

        legal_moves = MoveGenerator(self.board).generate_legal_moves(color)
        if not legal_moves:
            return None

        if color == WHITE:
            best_score = -INF
            for move in legal_moves:
                if self.board.make_move(move):
                    score = self._minimax(self.max_depth - 1, False)
                    self.board.undo_move(move)
                    if score > best_score:
                        best_score = score
                        self.best_move = move
        else:
            best_score = INF
            for move in legal_moves:
                if self.board.make_move(move):
                    score = self._minimax(self.max_depth - 1, True)
                    self.board.undo_move(move)
                    if score < best_score:
                        best_score = score
                        self.best_move = move

        return self.best_move

    def _minimax(self, depth: int, is_maximizing: bool) -> int:
        """
        Recursive minimax evaluation.

        Args:
            depth: Remaining search depth
            is_maximizing: True if current player tries to maximize score

        Returns:
            Evaluation score from White's perspective
        """
        self.nodes_searched += 1

        # Terminal node: evaluate position
        if depth == 0:
            return self.evaluator.evaluate(self.board)

        color = WHITE if is_maximizing else BLACK
        mg = MoveGenerator(self.board)
        legal_moves = mg.generate_legal_moves(color)

        # Terminal conditions
        if not legal_moves:
            if self.board.is_in_check(color):
                # Checkmate: worst outcome for the side to move
                return -INF + (self.max_depth - depth) if is_maximizing else INF - (self.max_depth - depth)
            else:
                return 0  # Stalemate

        if is_maximizing:
            best = -INF
            for move in legal_moves:
                if self.board.make_move(move):
                    score = self._minimax(depth - 1, False)
                    self.board.undo_move(move)
                    best = max(best, score)
            return best
        else:
            best = INF
            for move in legal_moves:
                if self.board.make_move(move):
                    score = self._minimax(depth - 1, True)
                    self.board.undo_move(move)
                    best = min(best, score)
            return best

    def get_stats(self) -> dict:
        """Return search statistics."""
        return {
            'nodes_searched': self.nodes_searched,
            'depth': self.max_depth,
            'best_move': str(self.best_move) if self.best_move else None,
        }
