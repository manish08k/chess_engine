"""
ui/pygame_board.py
==================
Pygame Chess Board UI — pieces drawn as clean geometric shapes.

Every piece is built from simple layers (bottom → top):
  1. BASE  — wide flat rectangle at the bottom
  2. BODY  — the main shape of the piece
  3. TOP   — head / crown / cross on top
  4. DETAIL — eyes, gems, engraving lines

No external fonts or image files needed.
Scales with SQ (square size) using a unit multiplier u = SQ/80.
"""

import sys, os, math, time
import pygame
from typing import Optional, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from board import Board, Move, WHITE, BLACK, EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING
from move_generator import MoveGenerator

# ─────────────────────────────────────────────────────────
#  Layout constants
# ─────────────────────────────────────────────────────────
SQ       = 80          # pixels per square
BOARD_PX = SQ * 8      # 640
PANEL_W  = 270         # right side panel width
WIN_W    = BOARD_PX + PANEL_W
WIN_H    = BOARD_PX    # 640

# ─────────────────────────────────────────────────────────
#  Colour palette
# ─────────────────────────────────────────────────────────
C_LIGHT    = (240, 217, 181)   # light board square
C_DARK     = (181, 136,  99)   # dark board square
C_SEL      = ( 99, 175,  75)   # selected square (green)
C_LEGAL    = (106, 135,  75)   # legal-move dot
C_LAST_F   = (205, 210, 106)   # last-move FROM square
C_LAST_T   = (170, 180,  60)   # last-move TO square
C_CHECK    = (210,  40,  40)   # king-in-check red
C_BG       = ( 22,  15,   6)   # window background
C_PANEL    = ( 18,  12,   5)   # right panel background
C_BORDER   = ( 55,  38,  14)   # board border
C_GOLD     = (201, 168,  76)   # gold accent
C_GOLD_DIM = (100,  80,  30)   # dimmed gold divider
C_TEXT     = (232, 217, 184)   # panel text
C_DIM      = (110,  90,  50)   # dim label text

# Piece colours
WP_FILL = (255, 248, 220)   # white piece — ivory fill
WP_OUT  = ( 50,  32,   8)   # white piece — dark outline
BP_FILL = ( 26,  14,   2)   # black piece — near-black fill
BP_OUT  = (195, 155,  55)   # black piece — gold outline

# Lookup tables
PIECE_VAL     = {PAWN: 1, KNIGHT: 3, BISHOP: 3, ROOK: 5, QUEEN: 9}
PIECE_LETTERS = {PAWN:'P', KNIGHT:'N', BISHOP:'B', ROOK:'R', QUEEN:'Q', KING:'K'}


# ═════════════════════════════════════════════════════════
#  Primitive drawing helpers
#  All coordinates are floats; cast to int before drawing.
# ═════════════════════════════════════════════════════════

def _p(v) -> int:
    """Convert float coordinate to int pixel."""
    return int(round(v))

def filled_circle(surf, cx, cy, r, fill, outline, lw=2):
    """Draw a filled circle with an outline ring."""
    pygame.draw.circle(surf, fill,    (_p(cx), _p(cy)), max(1, _p(r)))
    pygame.draw.circle(surf, outline, (_p(cx), _p(cy)), max(1, _p(r)), max(1, lw))

def filled_rect(surf, x, y, w, h, fill, outline, lw=2):
    """Draw a filled rectangle with an outline."""
    pygame.draw.rect(surf, fill,    (_p(x), _p(y), max(1, _p(w)), max(1, _p(h))))
    pygame.draw.rect(surf, outline, (_p(x), _p(y), max(1, _p(w)), max(1, _p(h))), max(1, lw))

def filled_polygon(surf, pts, fill, outline, lw=2):
    """Draw a filled polygon with an outline."""
    ipts = [(_p(x), _p(y)) for x, y in pts]
    pygame.draw.polygon(surf, fill,    ipts)
    pygame.draw.polygon(surf, outline, ipts, max(1, lw))

def line(surf, x1, y1, x2, y2, colour, lw=2):
    pygame.draw.line(surf, colour, (_p(x1), _p(y1)), (_p(x2), _p(y2)), max(1, lw))


# ═════════════════════════════════════════════════════════
#  Individual piece drawing functions
#
#  Each function receives:
#    surf  — pygame.Surface to draw onto
#    cx,cy — centre of the square in pixels
#    u     — scale unit  (u=1.0 when SQ=80;  u=0.75 when SQ=60)
#    F     — fill colour
#    O     — outline colour
#
#  Piece anatomy (top of screen = small row index):
#    cy-24u  ← very top (crown tip / king cross)
#    cy-16u  ← upper head area
#    cy-8u   ← lower head / neck
#    cy+0u   ← mid body
#    cy+8u   ← lower body
#    cy+16u  ← base top
#    cy+24u  ← base bottom  (SQ/2 = 40u from centre)
# ═════════════════════════════════════════════════════════

def draw_pawn(surf, cx, cy, u, F, O):
    """
    PAWN
    ────
       ( o )       ← round head
         |         ← thin neck
      [─────]      ← wide base
    """
    # 1. BASE — wide flat slab
    filled_rect(surf, cx - 16*u, cy + 14*u, 32*u, 9*u, F, O)
    # 2. NECK — thin vertical stem
    filled_rect(surf, cx - 5*u,  cy + 3*u,  10*u, 13*u, F, O)
    # 3. HEAD — large circle
    filled_circle(surf, cx, cy - 4*u, 13*u, F, O)


def draw_rook(surf, cx, cy, u, F, O):
    """
    ROOK  (castle / tower)
    ──────────────────────
     ▐█▌ ▐█▌ ▐█▌   ← three merlons (battlements)
     [       ]     ← wide tower top
     [       ]     ← tower body
    [─────────]    ← base
    """
    # 1. BASE
    filled_rect(surf, cx - 20*u, cy + 14*u, 40*u, 9*u, F, O)
    # 2. TOWER BODY
    filled_rect(surf, cx - 13*u, cy - 2*u,  26*u, 18*u, F, O)
    # 3. MERLON BAR — solid bar just above body top
    filled_rect(surf, cx - 13*u, cy - 8*u,  26*u,  8*u, F, O)
    # 4. THREE MERLONS — individual raised blocks
    for offset in (-9*u, 0, 9*u):
        filled_rect(surf, cx + offset - 4*u, cy - 20*u, 9*u, 14*u, F, O)


def draw_knight(surf, cx, cy, u, F, O):
    """
    KNIGHT  (horse head, facing right)
    ────────────────────────────────────
          ___
         /   \\←  forehead
        | (•) |  ← eye
        |     |
        |_____|
       /snout  \\  ← jaw / muzzle
      [─────────] ← base
    """
    # 1. BASE
    filled_rect(surf, cx - 18*u, cy + 14*u, 36*u, 9*u, F, O)

    # 2. NECK / LOWER BODY — angled trapezoid
    neck = [
        (cx - 11*u, cy + 14*u),   # bottom-left
        (cx + 13*u, cy + 14*u),   # bottom-right
        (cx + 14*u, cy +  2*u),   # right side
        (cx -  8*u, cy +  2*u),   # left side
    ]
    filled_polygon(surf, neck, F, O)

    # 3. HEAD — main horse-head silhouette
    head = [
        (cx -  8*u, cy +  2*u),   # neck-left top
        (cx + 14*u, cy +  2*u),   # neck-right top
        (cx + 17*u, cy - 10*u),   # forehead right
        (cx + 13*u, cy - 20*u),   # top of head
        (cx +  2*u, cy - 24*u),   # poll (very top)
        (cx -  8*u, cy - 20*u),   # back of head
        (cx - 12*u, cy - 10*u),   # back of neck
    ]
    filled_polygon(surf, head, F, O)

    # 4. MUZZLE — small forward-protruding box
    muzzle = [
        (cx + 14*u, cy +  2*u),
        (cx + 20*u, cy -  4*u),
        (cx + 20*u, cy - 12*u),
        (cx + 14*u, cy - 12*u),
        (cx + 11*u, cy -  6*u),
    ]
    filled_polygon(surf, muzzle, F, O)

    # 5. EYE — solid dark dot
    pygame.draw.circle(surf, O, (_p(cx + 8*u), _p(cy - 15*u)), max(2, _p(3.5*u)))

    # 6. NOSTRIL — tiny dot on muzzle
    pygame.draw.circle(surf, O, (_p(cx + 18*u), _p(cy - 6*u)), max(1, _p(2*u)))

    # 7. MANE LINE — decorative stroke along back of head
    line(surf, cx - 10*u, cy - 8*u,  cx - 4*u, cy - 22*u, O, max(1, _p(1.5*u)))


def draw_bishop(surf, cx, cy, u, F, O):
    """
    BISHOP  (tall mitre hat)
    ────────────────────────
          •          ← finial ball
         /|\\         ← mitre tip
        ( O )        ← head with cross engraved
       /     \\       ← upper body taper
      [───────]      ← base
    """
    # 1. BASE
    filled_rect(surf, cx - 18*u, cy + 14*u, 36*u, 9*u, F, O)

    # 2. LOWER BODY — wide trapezoid
    lower = [
        (cx - 15*u, cy + 14*u),
        (cx + 15*u, cy + 14*u),
        (cx +  9*u, cy +  2*u),
        (cx -  9*u, cy +  2*u),
    ]
    filled_polygon(surf, lower, F, O)

    # 3. UPPER BODY — narrowing trapezoid
    upper = [
        (cx -  9*u, cy +  3*u),
        (cx +  9*u, cy +  3*u),
        (cx +  5*u, cy -  8*u),
        (cx -  5*u, cy -  8*u),
    ]
    filled_polygon(surf, upper, F, O)

    # 4. HEAD — circle (the round part of the mitre)
    filled_circle(surf, cx, cy - 13*u, 9*u, F, O)

    # 5. MITRE POINT — small triangle spike on top
    mitre = [
        (cx - 3*u, cy - 20*u),
        (cx + 3*u, cy - 20*u),
        (cx,       cy - 27*u),
    ]
    filled_polygon(surf, mitre, F, O)

    # 6. FINIAL — tiny ball at very top
    filled_circle(surf, cx, cy - 28*u, 3*u, F, O)

    # 7. CROSS — engraved on the head circle
    line(surf, cx,       cy - 18*u, cx,       cy -  9*u, O, max(1, _p(1.8*u)))
    line(surf, cx - 5*u, cy - 14*u, cx + 5*u, cy - 14*u, O, max(1, _p(1.8*u)))


def draw_queen(surf, cx, cy, u, F, O):
    """
    QUEEN  (tall crown with 5 orbs)
    ────────────────────────────────
      • • • • •      ← 5 crown orbs
      [─────────]    ← crown band
     /           \\   ← body taper
    [─────────────]  ← base
    """
    # 1. BASE — widest piece on board
    filled_rect(surf, cx - 22*u, cy + 14*u, 44*u, 9*u, F, O)

    # 2. BODY — wide tapered trapezoid
    body = [
        (cx - 18*u, cy + 14*u),
        (cx + 18*u, cy + 14*u),
        (cx + 11*u, cy -  2*u),
        (cx - 11*u, cy -  2*u),
    ]
    filled_polygon(surf, body, F, O)

    # 3. CROWN BAND — horizontal bar
    filled_rect(surf, cx - 13*u, cy - 8*u, 26*u, 8*u, F, O)

    # 4. FIVE CROWN ORBS across the top
    orb_positions = [-10*u, -5*u, 0, 5*u, 10*u]
    for ox in orb_positions:
        filled_circle(surf, cx + ox, cy - 16*u, 4.5*u, F, O)

    # 5. CENTRE GEM — contrasting filled dot on body
    pygame.draw.circle(surf, O, (_p(cx), _p(cy - 2*u)), max(2, _p(3.5*u)))


def draw_king(surf, cx, cy, u, F, O):
    """
    KING  (crown with cross on top)
    ────────────────────────────────
           |           ← vertical cross bar
         ──+──         ← horizontal cross bar
      [─────────]      ← crown band
     /           \\     ← body taper
    [─────────────]    ← base
    """
    # 1. BASE
    filled_rect(surf, cx - 22*u, cy + 14*u, 44*u, 9*u, F, O)

    # 2. BODY — same taper as queen
    body = [
        (cx - 18*u, cy + 14*u),
        (cx + 18*u, cy + 14*u),
        (cx + 11*u, cy -  2*u),
        (cx - 11*u, cy -  2*u),
    ]
    filled_polygon(surf, body, F, O)

    # 3. CROWN BAND
    filled_rect(surf, cx - 13*u, cy - 8*u, 26*u, 8*u, F, O)

    # 4. CROSS — vertical shaft
    filled_rect(surf, cx - 3.5*u, cy - 26*u, 7*u, 20*u, F, O)

    # 5. CROSS — horizontal arm
    filled_rect(surf, cx - 10*u, cy - 21*u, 20*u, 6*u, F, O)


# ─────────────────────────────────────────────────────────
#  Dispatcher + Surface Cache
# ─────────────────────────────────────────────────────────

_DRAW_FN = {
    PAWN:   draw_pawn,
    KNIGHT: draw_knight,
    BISHOP: draw_bishop,
    ROOK:   draw_rook,
    QUEEN:  draw_queen,
    KING:   draw_king,
}

_surface_cache: dict = {}

def get_piece_surface(piece: int, sq_size: int) -> pygame.Surface:
    """
    Return a cached transparent surface with the piece drawn on it.
    Cache key = (piece, sq_size) so white/black + all 6 types are separate.
    """
    key = (piece, sq_size)
    if key in _surface_cache:
        return _surface_cache[key]

    surf = pygame.Surface((sq_size, sq_size), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))   # fully transparent

    is_white = piece > 0
    F = WP_FILL if is_white else BP_FILL
    O = WP_OUT  if is_white else BP_OUT
    u = sq_size / 80.0        # scale unit

    cx = sq_size / 2
    cy = sq_size / 2

    fn = _DRAW_FN.get(abs(piece))
    if fn:
        fn(surf, cx, cy, u, F, O)

    _surface_cache[key] = surf
    return surf


# ═════════════════════════════════════════════════════════
#  Piece animation (smooth slide)
# ═════════════════════════════════════════════════════════

class AnimatedPiece:
    """Lerps a piece surface from one pixel position to another."""

    def __init__(self, piece: int, from_px: Tuple, to_px: Tuple,
                 from_sq_idx: int, duration: float = 0.20):
        self.piece       = piece
        self.fx, self.fy = from_px
        self.tx, self.ty = to_px
        self.from_sq     = from_sq_idx   # row*8+col — skip drawing here
        self.t0          = time.time()
        self.duration    = duration
        self.done        = False

    def current_pos(self) -> Tuple[float, float]:
        """Ease-out cubic interpolation. Sets self.done when complete."""
        t = min(1.0, (time.time() - self.t0) / self.duration)
        t = 1 - (1 - t) ** 3           # ease-out cubic
        self.done = (t >= 1.0)
        return (self.fx + (self.tx - self.fx) * t,
                self.fy + (self.ty - self.fy) * t)


# ═════════════════════════════════════════════════════════
#  ChessBoardUI — main UI class
# ═════════════════════════════════════════════════════════

class ChessBoardUI:
    """
    Manages the pygame window, board rendering, piece animations,
    user input (click-to-move), and the side panel.
    """

    def __init__(self, board: Board, human_color: int = WHITE):
        pygame.init()
        pygame.display.set_caption("Chess AI Engine")
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.clock  = pygame.time.Clock()
        self.board  = board
        self.human  = human_color

        # Load fonts (ASCII only — always renders correctly)
        def font(names: str, size: int, bold=False):
            return pygame.font.SysFont(names, size, bold=bold)

        self.fnt_title = font("georgia,serif",           22, bold=True)
        self.fnt_ui    = font("consolas,couriernew",     17, bold=True)
        self.fnt_small = font("consolas,couriernew",     13)
        self.fnt_coord = font("consolas,couriernew",     11)

        # ── UI state ──────────────────────────────────────
        self.selected:      Optional[Tuple[int, int]] = None
        self.legal_targets: List[Move] = []
        self.last_move:     Optional[Move] = None
        self.thinking       = False
        self.think_angle    = 0.0
        self.status_msg     = ""
        self.move_log:      List[str] = []
        self.captured_w:    List[int] = []   # pieces captured by White
        self.captured_b:    List[int] = []   # pieces captured by Black
        self.eval_cp        = 0              # engine eval in centipawns
        self.promotion_pending: Optional[Tuple] = None
        self._promo_rects:  List[Tuple] = []
        self.anim:          Optional[AnimatedPiece] = None

        # Pre-render the static board texture
        self._board_surf = self._build_board_surface()

    # ─────────────────────────────────────────────────────
    #  Static board texture
    # ─────────────────────────────────────────────────────

    def _build_board_surface(self) -> pygame.Surface:
        """Draw the 8×8 chequered pattern once into a surface."""
        s = pygame.Surface((BOARD_PX, BOARD_PX))
        for r in range(8):
            for c in range(8):
                colour = C_LIGHT if (r + c) % 2 == 0 else C_DARK
                pygame.draw.rect(s, colour, (c * SQ, r * SQ, SQ, SQ))
        return s

    # ─────────────────────────────────────────────────────
    #  Master render call (called every frame from main.py)
    # ─────────────────────────────────────────────────────

    def render(self):
        self.screen.fill(C_BG)
        self._draw_board_and_highlights()
        self._draw_legal_move_dots()
        self._draw_all_pieces()
        self._draw_animated_piece()
        self._draw_rank_file_labels()
        self._draw_side_panel()
        if self.promotion_pending:
            self._draw_promotion_dialog()
        pygame.display.flip()

    # ─────────────────────────────────────────────────────
    #  Board + square highlights
    # ─────────────────────────────────────────────────────

    def _draw_board_and_highlights(self):
        # Drop-shadow under board
        shadow = pygame.Surface((BOARD_PX + 10, BOARD_PX + 10), pygame.SRCALPHA)
        pygame.draw.rect(shadow, (0, 0, 0, 100), (5, 5, BOARD_PX, BOARD_PX))
        self.screen.blit(shadow, (-3, -3))

        # Static checkerboard
        self.screen.blit(self._board_surf, (0, 0))

        # ── Last-move highlight (yellow tint) ────────────
        if self.last_move:
            for sq_pos, col in [(self.last_move.from_sq, C_LAST_F),
                                  (self.last_move.to_sq,   C_LAST_T)]:
                r, c = sq_pos
                hl = pygame.Surface((SQ, SQ), pygame.SRCALPHA)
                hl.fill((*col, 145))
                self.screen.blit(hl, (c * SQ, r * SQ))

        # ── Selected-piece highlight (green tint) ────────
        if self.selected:
            r, c = self.selected
            hl = pygame.Surface((SQ, SQ), pygame.SRCALPHA)
            hl.fill((*C_SEL, 185))
            self.screen.blit(hl, (c * SQ, r * SQ))

        # ── King-in-check highlight (pulsing red) ────────
        if self.board.is_in_check(self.board.turn):
            kr, kc = self.board.king_positions.get(self.board.turn, (-1, -1))
            if kr >= 0:
                alpha = int(80 + 70 * math.sin(time.time() * 5))
                hl = pygame.Surface((SQ, SQ), pygame.SRCALPHA)
                hl.fill((*C_CHECK, alpha))
                self.screen.blit(hl, (kc * SQ, kr * SQ))

        # Board border
        pygame.draw.rect(self.screen, C_BORDER, (0, 0, BOARD_PX, BOARD_PX), 2)

    # ─────────────────────────────────────────────────────
    #  Legal-move indicators
    # ─────────────────────────────────────────────────────

    def _draw_legal_move_dots(self):
        """
        Quiet moves  → small dark dot in the centre of the square.
        Capture moves → dark ring around the edge of the square.
        """
        if not self.selected:
            return
        for mv in self.legal_targets:
            tr, tc = mv.to_sq
            dot = pygame.Surface((SQ, SQ), pygame.SRCALPHA)
            is_capture = mv.captured != EMPTY or mv.is_en_passant
            if is_capture:
                pygame.draw.circle(dot, (0, 0, 0, 65),
                                    (SQ // 2, SQ // 2), SQ // 2 - 4, 6)
            else:
                pygame.draw.circle(dot, (0, 0, 0, 65),
                                    (SQ // 2, SQ // 2), SQ // 5)
            self.screen.blit(dot, (tc * SQ, tr * SQ))

    # ─────────────────────────────────────────────────────
    #  Piece rendering
    # ─────────────────────────────────────────────────────

    def _draw_all_pieces(self):
        """Blit every piece except the one currently being animated."""
        skip_idx = self.anim.from_sq if (self.anim and not self.anim.done) else -1
        for r in range(8):
            for c in range(8):
                p = int(self.board.squares[r][c])
                if p == EMPTY:
                    continue
                if r * 8 + c == skip_idx:
                    continue
                surf = get_piece_surface(p, SQ)
                self.screen.blit(surf, (c * SQ, r * SQ))

    def _draw_animated_piece(self):
        """Draw the smoothly-sliding piece during its animation."""
        if self.anim:
            x, y = self.anim.current_pos()
            surf  = get_piece_surface(self.anim.piece, SQ)
            self.screen.blit(surf, (_p(x), _p(y)))
            if self.anim.done:
                self.anim = None

    # ─────────────────────────────────────────────────────
    #  Rank / file coordinate labels
    # ─────────────────────────────────────────────────────

    def _draw_rank_file_labels(self):
        """
        Files a–h printed in the bottom-right corner of row 7.
        Ranks 8–1 printed in the top-left corner of column 0.
        Label colour alternates so it contrasts with its square.
        """
        files = 'abcdefgh'
        for i in range(8):
            # File label (bottom-right of each square in rank 7)
            file_col = C_DARK  if i % 2 == 0 else C_LIGHT
            fl = self.fnt_coord.render(files[i], True, file_col)
            self.screen.blit(fl, (i * SQ + SQ - fl.get_width() - 3,
                                    7 * SQ + SQ - fl.get_height() - 2))
            # Rank label (top-left of each square in file 0)
            rank_col = C_LIGHT if i % 2 == 0 else C_DARK
            rl = self.fnt_coord.render(str(8 - i), True, rank_col)
            self.screen.blit(rl, (3, i * SQ + 3))

    # ─────────────────────────────────────────────────────
    #  Side Panel
    # ─────────────────────────────────────────────────────

    def _draw_side_panel(self):
        px = BOARD_PX
        pygame.draw.rect(self.screen, C_PANEL, (px, 0, PANEL_W, WIN_H))
        pygame.draw.line(self.screen, C_BORDER,   (px,     0), (px,     WIN_H), 2)
        pygame.draw.line(self.screen, C_GOLD_DIM, (px + 1, 0), (px + 1, WIN_H), 1)

        x  = px + 14
        rw = PANEL_W - 28    # usable row width
        y  = 14

        def divider():
            nonlocal y
            pygame.draw.rect(self.screen, C_BORDER, (x, y, rw, 1))
            y += 9

        def label(txt):
            nonlocal y
            self.screen.blit(self.fnt_small.render(txt, True, C_DIM), (x, y))
            y += 16

        # ── Title ─────────────────────────────────────────
        t = self.fnt_title.render("Chess AI Engine", True, C_GOLD)
        self.screen.blit(t, (x, y)); y += 34
        divider()

        # ── Turn & status ──────────────────────────────────
        turn_str = ("White" if self.board.turn == WHITE else "Black") + "'s turn"
        tcol = (245, 240, 205) if self.board.turn == WHITE else (210, 155, 70)
        self.screen.blit(self.fnt_ui.render(turn_str, True, tcol), (x, y)); y += 26

        if self.thinking:
            self._draw_spinner(px + PANEL_W - 28, y - 13)

        if self.status_msg:
            sc = (220, 60, 60) if any(w in self.status_msg.upper()
                                       for w in ("CHECK", "MATE", "DRAW", "STALE")) else C_GOLD
            self.screen.blit(self.fnt_small.render(self.status_msg, True, sc), (x, y))
            y += 20

        y += 4; divider()

        # ── Evaluation bar ─────────────────────────────────
        label("EVALUATION")
        bh = 14
        pygame.draw.rect(self.screen, (30, 20, 8), (x, y, rw, bh), border_radius=7)
        pct = min(0.92, max(0.08, 0.5 + self.eval_cp / 3000.0))
        pygame.draw.rect(self.screen, (230, 220, 190),
                          (x, y, int(rw * pct), bh), border_radius=7)
        pygame.draw.rect(self.screen, C_BORDER, (x, y, rw, bh), 1, border_radius=7)
        ev_s = self.fnt_small.render(f"{self.eval_cp / 100:+.1f}", True, C_GOLD)
        self.screen.blit(ev_s, (x + rw + 4, y - 1))
        y += bh + 8

        # Material count
        wm = sum(PIECE_VAL.get(abs(int(self.board.squares[r2][c2])), 0)
                 for r2 in range(8) for c2 in range(8)
                 if int(self.board.squares[r2][c2]) > 0)
        bm = sum(PIECE_VAL.get(abs(int(self.board.squares[r2][c2])), 0)
                 for r2 in range(8) for c2 in range(8)
                 if int(self.board.squares[r2][c2]) < 0)
        diff = wm - bm
        ms = self.fnt_small.render(
            f"Material: {'+' if diff > 0 else ''}{diff}  "
            f"({'White' if diff > 0 else 'Black' if diff < 0 else 'Equal'})",
            True, C_TEXT)
        self.screen.blit(ms, (x, y)); y += 20

        y += 4; divider()

        # ── Captured pieces ────────────────────────────────
        label("CAPTURED PIECES")
        for side_label, pieces in [("By White:", self.captured_w),
                                     ("By Black:", self.captured_b)]:
            self.screen.blit(self.fnt_small.render(side_label, True, C_DIM), (x, y))
            y += 14
            row = "  ".join(PIECE_LETTERS.get(abs(p), '?') for p in pieces) if pieces else "--"
            self.screen.blit(self.fnt_small.render(row, True, C_TEXT), (x + 4, y))
            y += 16

        y += 4; divider()

        # ── Move log ───────────────────────────────────────
        label("MOVES")
        log_h   = WIN_H - y - 44
        max_vis = max(1, log_h // 16)
        pairs   = [(self.move_log[i],
                    self.move_log[i + 1] if i + 1 < len(self.move_log) else "")
                   for i in range(0, len(self.move_log), 2)]
        visible = pairs[max(0, len(pairs) - max_vis):]
        latest  = self.move_log[-1] if self.move_log else ""

        for idx, (wm2, bm2) in enumerate(visible):
            n   = max(0, len(pairs) - max_vis) + idx + 1
            num = self.fnt_small.render(f"{n}.", True, C_DIM)
            wc  = C_GOLD if wm2 == latest else C_TEXT
            bc  = C_GOLD if bm2 == latest else C_TEXT
            self.screen.blit(num,                                    (x,       y))
            self.screen.blit(self.fnt_small.render(wm2, True, wc),  (x + 30,  y))
            self.screen.blit(self.fnt_small.render(bm2, True, bc),  (x + 110, y))
            y += 16

        # ── Controls ───────────────────────────────────────
        y = WIN_H - 36
        divider()
        hint = self.fnt_small.render("[R] New Game    [F] Flip    [ESC] Quit", True, C_DIM)
        self.screen.blit(hint, (x, y))

    def _draw_spinner(self, cx: int, cy: int):
        """Animated spinning dots shown while the AI is thinking."""
        self.think_angle = (self.think_angle + 5) % 360
        for i in range(8):
            angle = math.radians(self.think_angle + i * 45)
            dx = int(cx + 10 * math.cos(angle))
            dy = int(cy + 10 * math.sin(angle))
            alpha = max(30, 255 - i * 30)
            dot = pygame.Surface((6, 6), pygame.SRCALPHA)
            pygame.draw.circle(dot, (*C_GOLD, alpha), (3, 3), 3)
            self.screen.blit(dot, (dx - 3, dy - 3))

    # ─────────────────────────────────────────────────────
    #  Promotion dialog
    # ─────────────────────────────────────────────────────

    def _draw_promotion_dialog(self):
        """Show a 4-button dialog to pick the promotion piece."""
        color = self.promotion_pending[2]

        # Semi-transparent overlay
        ov = pygame.Surface((BOARD_PX, WIN_H), pygame.SRCALPHA)
        ov.fill((0, 0, 0, 160))
        self.screen.blit(ov, (0, 0))

        # Dialog box
        dw, dh = 380, 155
        dx = (BOARD_PX - dw) // 2
        dy = (WIN_H    - dh) // 2
        pygame.draw.rect(self.screen, (28, 18, 8), (dx, dy, dw, dh), border_radius=12)
        pygame.draw.rect(self.screen, C_GOLD,      (dx, dy, dw, dh), 2, border_radius=12)

        title = self.fnt_ui.render("Promote Pawn — Choose Piece", True, C_GOLD)
        self.screen.blit(title, (dx + (dw - title.get_width()) // 2, dy + 10))

        choices = [(QUEEN, "Queen"), (ROOK, "Rook"), (BISHOP, "Bishop"), (KNIGHT, "Knight")]
        self._promo_rects = []
        mx, my = pygame.mouse.get_pos()

        for i, (pt, name) in enumerate(choices):
            bx, by, bw, bh = dx + 12 + i * 89, dy + 48, 82, 90
            hovered = bx <= mx <= bx + bw and by <= my <= by + bh
            pygame.draw.rect(self.screen,
                              (55, 38, 14) if hovered else (30, 18, 5),
                              (bx, by, bw, bh), border_radius=8)
            pygame.draw.rect(self.screen,
                              C_GOLD if hovered else C_BORDER,
                              (bx, by, bw, bh), 2, border_radius=8)

            # Small piece preview (56 px)
            ps = get_piece_surface(color * pt, 56)
            self.screen.blit(ps, (bx + (bw - 56) // 2, by + 4))

            # Piece name label
            lbl = self.fnt_coord.render(name, True, C_GOLD if hovered else C_DIM)
            self.screen.blit(lbl, (bx + (bw - lbl.get_width()) // 2, by + 74))

            self._promo_rects.append((pygame.Rect(bx, by, bw, bh), pt))

    # ─────────────────────────────────────────────────────
    #  Input handling
    # ─────────────────────────────────────────────────────

    def handle_click(self, pos: Tuple) -> Optional[Move]:
        """
        Process a mouse click.
        Returns a fully-validated Move if one was completed, else None.
        """
        x, y = pos

        # Promotion dialog takes priority
        if self.promotion_pending:
            for rect, pt in self._promo_rects:
                if rect.collidepoint(pos):
                    return self._complete_promotion(pt)
            return None

        # Ignore clicks on the panel
        if x >= BOARD_PX or self.board.turn != self.human:
            return None

        clicked = (y // SQ, x // SQ)   # (row, col)

        if self.selected is not None:
            # Try to complete a move to the clicked square
            mv = next((m for m in self.legal_targets if m.to_sq == clicked), None)
            if mv:
                if mv.promotion != EMPTY:
                    # Hold the move until the player picks a promotion piece
                    self.promotion_pending = (self.selected, clicked, self.human)
                    self._clear_selection()
                    return None
                self._clear_selection()
                return mv
            else:
                # Re-select if clicking another own piece
                p = int(self.board.squares[clicked[0]][clicked[1]])
                if p != EMPTY and (WHITE if p > 0 else BLACK) == self.human:
                    self._select(clicked)
                else:
                    self._clear_selection()
        else:
            self._select(clicked)

        return None

    def _select(self, sq2: Tuple[int, int]):
        """Select a piece and compute its legal destinations."""
        r, c = sq2
        p = int(self.board.squares[r][c])
        if p == EMPTY or (WHITE if p > 0 else BLACK) != self.human:
            self._clear_selection()
            return
        self.selected = sq2
        mg = MoveGenerator(self.board)
        self.legal_targets = [m for m in mg.generate_legal_moves(self.human)
                               if m.from_sq == sq2]

    def _clear_selection(self):
        self.selected = None
        self.legal_targets = []

    def _complete_promotion(self, piece_type: int) -> Optional[Move]:
        """Match the chosen promotion piece to a legal move."""
        if not self.promotion_pending:
            return None
        from_sq, to_sq, color = self.promotion_pending
        self.promotion_pending = None
        mg = MoveGenerator(self.board)
        for mv in mg.generate_legal_moves(color):
            if (mv.from_sq == from_sq and mv.to_sq == to_sq
                    and abs(mv.promotion) == piece_type):
                return mv
        return None

    # ─────────────────────────────────────────────────────
    #  Public state setters (called from main.py)
    # ─────────────────────────────────────────────────────

    def start_anim(self, move: Move):
        """Kick off a slide animation for the given move."""
        fr, fc = move.from_sq
        tr, tc = move.to_sq
        piece   = int(self.board.squares[fr][fc])
        self.anim = AnimatedPiece(
            piece,
            from_px     = (fc * SQ, fr * SQ),
            to_px       = (tc * SQ, tr * SQ),
            from_sq_idx = fr * 8 + fc,
        )

    def on_move_made(self, move: Move, san: str = ""):
        """Update UI state after a move has been applied to the board."""
        self.last_move = move
        self._clear_selection()
        if san:
            self.move_log.append(san)

    def set_status(self, msg: str):   self.status_msg = msg
    def set_thinking(self, on: bool): self.thinking   = on
    def set_eval(self, cp: int):      self.eval_cp    = cp

    # ─────────────────────────────────────────────────────
    #  SAN generator (Simple Algebraic Notation)
    # ─────────────────────────────────────────────────────

    def move_to_san(self, move: Move, board_before: Board) -> str:
        """Convert a Move to short algebraic notation (e.g. Nf3, exd5, O-O)."""
        files = 'abcdefgh'
        fr, fc = move.from_sq
        tr, tc = move.to_sq
        p  = int(board_before.squares[fr][fc])
        ap = abs(p)
        piece_letter = {KNIGHT:'N', BISHOP:'B', ROOK:'R', QUEEN:'Q', KING:'K'}

        if move.is_castling:
            return "O-O" if tc == 6 else "O-O-O"

        letter     = piece_letter.get(ap, '')
        capture    = 'x' if move.captured != EMPTY or move.is_en_passant else ''
        from_file  = files[fc] if ap == PAWN and capture else ''
        dest       = files[tc] + str(8 - tr)
        promotion  = ('=' + piece_letter.get(abs(move.promotion), 'Q')) if move.promotion else ''

        return f"{letter}{from_file}{capture}{dest}{promotion}"