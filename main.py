"""
main.py - Chess AI Engine  (v3 — fixed turn alternation)

Turn flow:
  HUMAN_TURN  → human clicks a piece and destination
              → move applied → switches to AI_TURN
  AI_TURN     → AI computes in background thread
              → result collected → move applied → switches to HUMAN_TURN
  GAME_OVER   → show result, wait for R to restart

Controls:  click to move | R = restart | F = flip | ESC = quit
"""

import sys, os, time, threading, copy
from typing import Optional
import pygame

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from board import Board, Move, WHITE, BLACK
from move_generator import MoveGenerator
from evaluation import Evaluator
from alpha_beta import AlphaBetaSearch
from opening_book import OpeningBook
from ui.pygame_board import ChessBoardUI

# ── Config ────────────────────────────────────────────────
HUMAN_COLOR      = WHITE
AI_COLOR         = BLACK
AI_DEPTH         = 3          # lower = faster response (3-4 recommended)
AI_TIME_LIMIT    = 5.0        # seconds
USE_OPENING_BOOK = True
FPS              = 60
DATA_DIR         = os.path.join(BASE_DIR, "data")

# ── Game states ───────────────────────────────────────────
STATE_HUMAN    = "human"
STATE_AI       = "ai"
STATE_GAMEOVER = "gameover"


class ChessGame:
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)

        self.board  = Board()
        self.ui     = ChessBoardUI(self.board, human_color=HUMAN_COLOR)
        self.evaluator = Evaluator()

        # Opening book
        try:
            self.opening_book = OpeningBook(os.path.join(DATA_DIR, "opening_book.pkl"))
        except Exception:
            self.opening_book = None

        # State machine
        self.state      = STATE_HUMAN   # always starts as human (white)
        self.ai_thread: Optional[threading.Thread] = None
        self.ai_move:   Optional[Move]  = None   # result from AI thread
        self.ai_error   = False

    # ═════════════════════════════════════════════════════
    # Main Loop
    # ═════════════════════════════════════════════════════
    def run(self):
        print("="*48)
        print("  Chess AI  |  You=White  AI=Black  |  Depth", AI_DEPTH)
        print("  R=Restart  F=Flip  ESC=Quit")
        print("="*48)

        clock = pygame.time.Clock()

        while True:
            # ── Events ───────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit(); sys.exit()
                    elif event.key == pygame.K_r:
                        self._restart()
                    elif event.key == pygame.K_f:
                        pass  # flip is visual only, handled in ui

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.state == STATE_HUMAN:
                        self._on_human_click(event.pos)

            # ── State transitions ─────────────────────────
            if self.state == STATE_AI:
                self._tick_ai()

            # ── Eval bar (cheap, runs every frame) ───────
            self.ui.set_eval(self.evaluator.evaluate(self.board))

            # ── Render ────────────────────────────────────
            self.ui.render()
            clock.tick(FPS)

    # ═════════════════════════════════════════════════════
    # Human Turn
    # ═════════════════════════════════════════════════════
    def _on_human_click(self, pos):
        """Process a mouse click during the human's turn."""
        move = self.ui.handle_click(pos)
        if move is None:
            return  # selecting piece or invalid click

        self._execute_move(move, by_human=True)

        if self.state != STATE_GAMEOVER:
            # Hand off to AI
            self.state = STATE_AI
            self._start_ai_thread()

    # ═════════════════════════════════════════════════════
    # AI Turn
    # ═════════════════════════════════════════════════════
    def _start_ai_thread(self):
        """Kick off AI computation in background."""
        self.ai_move  = None
        self.ai_error = False
        self.ui.set_thinking(True)
        self.ui.set_status("AI is thinking...")

        # Pass a SNAPSHOT of the board so the thread can't race with main thread
        board_snapshot = self.board.copy()

        self.ai_thread = threading.Thread(
            target=self._ai_worker,
            args=(board_snapshot,),
            daemon=True
        )
        self.ai_thread.start()

    def _ai_worker(self, board_snapshot: Board):
        """Runs in background thread. Writes result to self.ai_move."""
        try:
            # Try opening book first
            if self.opening_book:
                bm = self.opening_book.get_move(board_snapshot)
                if bm:
                    time.sleep(0.3)   # brief pause so it feels natural
                    self.ai_move = bm
                    return

            # Alpha-beta search on the snapshot
            search = AlphaBetaSearch(
                board_snapshot,
                max_depth=AI_DEPTH,
                time_limit=AI_TIME_LIMIT
            )
            mv = search.search()
            st = search.get_stats()
            print(f"  [AI] {mv} | depth={st['depth']} "
                  f"nodes={st['nodes']:,} time={st['time_ms']}ms")
            self.ai_move = mv

        except Exception as e:
            import traceback
            print(f"[AI ERROR] {e}")
            traceback.print_exc()
            # Fallback: random legal move
            import random
            moves = MoveGenerator(board_snapshot).generate_legal_moves(AI_COLOR)
            self.ai_move = random.choice(moves) if moves else None
            self.ai_error = True

    def _tick_ai(self):
        """Called every frame while waiting for AI. Picks up result when ready."""
        # Still computing
        if self.ai_thread and self.ai_thread.is_alive():
            return

        # Thread finished — collect result
        self.ui.set_thinking(False)
        self.ui.set_status("")

        mv = self.ai_move
        self.ai_move   = None
        self.ai_thread = None

        if mv is None:
            # No move = AI has no legal moves (shouldn't normally happen here)
            print("[AI] No move returned — checking game over")
            self._check_game_over()
            return

        # Validate the move still applies to the REAL board
        # (board may have been restarted while AI was thinking)
        legal = MoveGenerator(self.board).generate_legal_moves(AI_COLOR)
        matched = next((m for m in legal
                        if m.from_sq == mv.from_sq and
                           m.to_sq   == mv.to_sq and
                           m.promotion == mv.promotion), None)

        if matched is None:
            print(f"[AI] Move {mv} no longer valid on real board — picking random")
            import random
            matched = random.choice(legal) if legal else None

        if matched:
            self._execute_move(matched, by_human=False)

        # Back to human if game still going
        if self.state != STATE_GAMEOVER:
            self.state = STATE_HUMAN

    # ═════════════════════════════════════════════════════
    # Execute Move (shared by human + AI)
    # ═════════════════════════════════════════════════════
    def _execute_move(self, move: Move, by_human: bool):
        """Apply move to board, update UI, check game over."""
        # Generate SAN before the move changes the board
        san = self.ui.move_to_san(move, self.board)

        # Track captured pieces for display
        if move.captured:
            if by_human:
                self.ui.captured_w.append(move.captured)
            else:
                self.ui.captured_b.append(move.captured)

        # Start visual animation
        self.ui.start_anim(move)

        # Apply to real board
        ok = self.board.make_move(move)
        if not ok:
            print(f"[!] make_move rejected: {move}")
            return

        self.ui.on_move_made(move, san)
        print(f"  [{'You' if by_human else 'AI '}] {move}  ({san})  "
              f"move={self.board.fullmove_number}")

        self._check_game_over()

    # ═════════════════════════════════════════════════════
    # Game Over Check
    # ═════════════════════════════════════════════════════
    def _check_game_over(self):
        color = self.board.turn
        mg    = MoveGenerator(self.board)
        moves = mg.generate_legal_moves(color)

        if not moves:
            self.state = STATE_GAMEOVER
            if self.board.is_in_check(color):
                winner = "Black wins!" if color == WHITE else "White wins!"
                self.ui.set_status(f"CHECKMATE — {winner}")
                print(f"  [GAME OVER] Checkmate — {winner}")
            else:
                self.ui.set_status("STALEMATE — Draw!")
                print("  [GAME OVER] Stalemate")

        elif self.board.is_draw_by_fifty_moves():
            self.state = STATE_GAMEOVER
            self.ui.set_status("DRAW — 50-move rule")

        elif self.board.is_insufficient_material():
            self.state = STATE_GAMEOVER
            self.ui.set_status("DRAW — Insufficient material")

        elif self.board.is_in_check(color):
            self.ui.set_status("Check!")

        else:
            self.ui.set_status("")

    # ═════════════════════════════════════════════════════
    # Restart
    # ═════════════════════════════════════════════════════
    def _restart(self):
        # Kill any running AI thread (it's daemon so it'll die anyway)
        self.ai_thread = None
        self.ai_move   = None

        self.board = Board()
        self.ui.board      = self.board
        self.ui.last_move  = None
        self.ui.move_log   = []
        self.ui.captured_w = []
        self.ui.captured_b = []
        self.ui.anim       = None
        self.ui.selected   = None
        self.ui.legal_targets = []
        self.ui.set_status("")
        self.ui.set_thinking(False)
        self.ui.set_eval(0)

        self.state = STATE_HUMAN
        print("  [RESTARTED]")


# ── Entry Point ───────────────────────────────────────────
def main():
    ChessGame().run()


if __name__ == "__main__":
    main()