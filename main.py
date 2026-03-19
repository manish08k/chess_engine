"""
main.py - Chess Engine  (Human vs AI)

Self-contained: search engine built right here so there are no
import issues between modules.  Drop this file into your repo
alongside board.py and move_generator.py — that is all you need.

Controls:
  Click piece  -> select (legal moves highlighted)
  Click square -> move
  R            -> restart
  F            -> flip board
  ESC          -> quit
"""

import threading
import time
import sys

# ── Configuration ────────────────────────────────────────────────────
HUMAN_COLOR      = 1      # 1 = play White,  -1 = play Black
AI_DEPTH         = 4      # search depth  (3=fast, 4=good, 5=strong)
AI_TIME_LIMIT    = 3.0    # max seconds per AI move
BOARD_FLIP       = False  # True = flip so human is always at bottom
# ─────────────────────────────────────────────────────────────────────

from board import Board, WHITE, BLACK, EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING
from move_generator import MoveGenerator

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

INF  = 10_000_000
MATE = 9_000_000

PIECE_VAL = {PAWN:100, KNIGHT:320, BISHOP:330, ROOK:500, QUEEN:900, KING:20000, EMPTY:0}

_PAWN_PST = [
  0,  0,  0,  0,  0,  0,  0,  0,
 50, 50, 50, 50, 50, 50, 50, 50,
 10, 10, 20, 30, 30, 20, 10, 10,
  5,  5, 10, 25, 25, 10,  5,  5,
  0,  0,  0, 20, 20,  0,  0,  0,
  5, -5,-10,  0,  0,-10, -5,  5,
  5, 10, 10,-20,-20, 10, 10,  5,
  0,  0,  0,  0,  0,  0,  0,  0,
]
_KNIGHT_PST = [
-50,-40,-30,-30,-30,-30,-40,-50,
-40,-20,  0,  0,  0,  0,-20,-40,
-30,  0, 10, 15, 15, 10,  0,-30,
-30,  5, 15, 20, 20, 15,  5,-30,
-30,  0, 15, 20, 20, 15,  0,-30,
-30,  5, 10, 15, 15, 10,  5,-30,
-40,-20,  0,  5,  5,  0,-20,-40,
-50,-40,-30,-30,-30,-30,-40,-50,
]
_BISHOP_PST = [
-20,-10,-10,-10,-10,-10,-10,-20,
-10,  0,  0,  0,  0,  0,  0,-10,
-10,  0,  5, 10, 10,  5,  0,-10,
-10,  5,  5, 10, 10,  5,  5,-10,
-10,  0, 10, 10, 10, 10,  0,-10,
-10, 10, 10, 10, 10, 10, 10,-10,
-10,  5,  0,  0,  0,  0,  5,-10,
-20,-10,-10,-10,-10,-10,-10,-20,
]
_ROOK_PST = [
  0,  0,  0,  0,  0,  0,  0,  0,
  5, 10, 10, 10, 10, 10, 10,  5,
 -5,  0,  0,  0,  0,  0,  0, -5,
 -5,  0,  0,  0,  0,  0,  0, -5,
 -5,  0,  0,  0,  0,  0,  0, -5,
 -5,  0,  0,  0,  0,  0,  0, -5,
 -5,  0,  0,  0,  0,  0,  0, -5,
  0,  0,  0,  5,  5,  0,  0,  0,
]
_QUEEN_PST = [
-20,-10,-10, -5, -5,-10,-10,-20,
-10,  0,  0,  0,  0,  0,  0,-10,
-10,  0,  5,  5,  5,  5,  0,-10,
 -5,  0,  5,  5,  5,  5,  0, -5,
  0,  0,  5,  5,  5,  5,  0, -5,
-10,  5,  5,  5,  5,  5,  0,-10,
-10,  0,  5,  0,  0,  0,  0,-10,
-20,-10,-10, -5, -5,-10,-10,-20,
]
_KING_PST = [
-30,-40,-40,-50,-50,-40,-40,-30,
-30,-40,-40,-50,-50,-40,-40,-30,
-30,-40,-40,-50,-50,-40,-40,-30,
-30,-40,-40,-50,-50,-40,-40,-30,
-20,-30,-30,-40,-40,-30,-30,-20,
-10,-20,-20,-20,-20,-20,-20,-10,
 20, 20,  0,  0,  0,  0, 20, 20,
 20, 30, 10,  0,  0, 10, 30, 20,
]

PST = {PAWN:_PAWN_PST, KNIGHT:_KNIGHT_PST, BISHOP:_BISHOP_PST,
       ROOK:_ROOK_PST, QUEEN:_QUEEN_PST,   KING:_KING_PST}

def _mirror(sq):
    r, c = sq//8, sq%8
    return (7-r)*8+c

def static_eval(board):
    score = 0
    for r in range(8):
        for c in range(8):
            p = int(board.squares[r][c])
            if p == 0: continue
            col = 1 if p>0 else -1
            pt  = abs(p)
            sq  = r*8+c
            tbl = PST.get(pt)
            pst = tbl[sq if col==1 else _mirror(sq)] if tbl else 0
            score += col*(PIECE_VAL.get(pt,0)+pst)
    return score

def order_moves(moves):
    def ms(m):
        if m.promotion != EMPTY: return 20000
        if m.captured  != EMPTY:
            return 10000 + PIECE_VAL.get(abs(m.captured),0)*10 - PIECE_VAL.get(abs(m.piece),0)
        return 0
    return sorted(moves, key=ms, reverse=True)


class Searcher:
    def __init__(self, board):
        self.board = board
        self.start = time.time()
        self.nodes = 0
        self.abort = False

    def elapsed(self):
        return time.time()-self.start

    def search(self):
        color = self.board.turn
        mg    = MoveGenerator(self.board)
        legal = mg.generate_legal_moves(color)
        if not legal: return None
        if len(legal)==1: return legal[0]

        best = legal[0]
        for depth in range(1, AI_DEPTH+1):
            if self.elapsed() >= AI_TIME_LIMIT: break
            self.abort = False
            score, move = self._root(depth, color)
            if not self.abort and move:
                best = move
            if self.abort: break
            print(f"  depth {depth}  score {score:+}  move {move}  nodes {self.nodes}  {self.elapsed():.2f}s")
            if abs(score) >= MATE-500: break
        return best

    def _root(self, depth, color):
        alpha, beta = -INF, INF
        best_score  = -INF
        best_move   = None
        mg    = MoveGenerator(self.board)
        moves = order_moves(mg.generate_legal_moves(color))
        for move in moves:
            if self.elapsed() >= AI_TIME_LIMIT:
                self.abort = True; break
            if self.board.make_move(move):
                score = -self._ab(depth-1, -beta, -alpha, -color)
                self.board.undo_move(move)
                if score > best_score:
                    best_score = score
                    best_move  = move
                    alpha      = max(alpha, score)
        return best_score, best_move

    def _ab(self, depth, alpha, beta, color):
        self.nodes += 1
        if self.nodes%2048==0 and self.elapsed()>=AI_TIME_LIMIT:
            self.abort = True
        if self.abort: return 0
        if self.board.is_draw_by_fifty_moves() or self.board.is_insufficient_material():
            return 0
        if depth <= 0:
            return self._qsearch(alpha, beta, color)
        mg    = MoveGenerator(self.board)
        moves = mg.generate_legal_moves(color)
        if not moves:
            return -(MATE-self.nodes%200) if self.board.is_in_check(color) else 0
        best = -INF
        for move in order_moves(moves):
            if not self.board.make_move(move): continue
            score = -self._ab(depth-1, -beta, -alpha, -color)
            self.board.undo_move(move)
            if self.abort: return 0
            best  = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta: break
        return best

    def _qsearch(self, alpha, beta, color):
        self.nodes += 1
        stand = static_eval(self.board)*color
        if stand >= beta: return beta
        alpha = max(alpha, stand)
        mg = MoveGenerator(self.board)
        caps = order_moves([m for m in mg.generate_legal_moves(color) if m.captured!=EMPTY])
        for move in caps:
            if self.board.make_move(move):
                score = -self._qsearch(-beta, -alpha, -color)
                self.board.undo_move(move)
                if score >= beta: return beta
                alpha = max(alpha, score)
        return alpha


def get_ai_move(board):
    try:
        s = Searcher(board)
        return s.search()
    except Exception as e:
        print(f"[Search error] {e}")
        import traceback; traceback.print_exc()
        legal = MoveGenerator(board).generate_legal_moves(board.turn)
        return legal[0] if legal else None


def draw_piece(surface, piece_type, is_white, x, y, sq):
    fill = (255,245,210) if is_white else (40,25,10)
    out  = (80,50,20)   if is_white else (210,170,60)
    cx   = x+sq//2
    bot  = y+sq-10
    u    = max(1, sq//10)

    def poly(pts):
        if len(pts)>=3:
            pygame.draw.polygon(surface, fill, pts)
            pygame.draw.polygon(surface, out,  pts, 2)

    def circ(cx2,cy,r):
        pygame.draw.circle(surface, fill, (cx2,cy), r)
        pygame.draw.circle(surface, out,  (cx2,cy), r, 2)

    def base():
        poly([(cx-3*u,bot),(cx+3*u,bot),(cx+2*u,bot-2*u),(cx-2*u,bot-2*u)])

    if piece_type==PAWN:
        base()
        poly([(cx-u,bot-2*u),(cx+u,bot-2*u),(cx+u,bot-4*u),(cx-u,bot-4*u)])
        circ(cx,bot-6*u,2*u)
    elif piece_type==KNIGHT:
        base()
        poly([(cx-2*u,bot-2*u),(cx-3*u,bot-6*u),(cx-u,bot-9*u),
              (cx+3*u,bot-9*u),(cx+3*u,bot-6*u),(cx+u,bot-3*u),(cx+2*u,bot-2*u)])
        pygame.draw.circle(surface, out, (cx+u,bot-7*u), max(1,u//2+1))
    elif piece_type==BISHOP:
        base()
        poly([(cx-2*u,bot-2*u),(cx+2*u,bot-2*u),(cx+u,bot-6*u),(cx-u,bot-6*u)])
        circ(cx,bot-7*u,u+1)
        pygame.draw.line(surface,out,(cx,bot-8*u),(cx,bot-10*u),2)
    elif piece_type==ROOK:
        base()
        poly([(cx-2*u,bot-2*u),(cx+2*u,bot-2*u),(cx+2*u,bot-6*u),(cx-2*u,bot-6*u)])
        for dx in [-2,0,2]:
            poly([(cx+dx*u-u,bot-6*u),(cx+dx*u+u,bot-6*u),
                  (cx+dx*u+u,bot-8*u),(cx+dx*u-u,bot-8*u)])
    elif piece_type==QUEEN:
        base()
        poly([(cx-2*u,bot-2*u),(cx+2*u,bot-2*u),(cx+u,bot-6*u),(cx-u,bot-6*u)])
        pygame.draw.rect(surface,fill,(cx-2*u,bot-7*u,4*u,u+1))
        pygame.draw.rect(surface,out, (cx-2*u,bot-7*u,4*u,u+1),1)
        for dx in [-2,-1,0,1,2]:
            circ(cx+dx*u,bot-8*u,max(1,u-1))
    elif piece_type==KING:
        base()
        poly([(cx-2*u,bot-2*u),(cx+2*u,bot-2*u),(cx+u,bot-6*u),(cx-u,bot-6*u)])
        pygame.draw.rect(surface,fill,(cx-2*u,bot-7*u,4*u,u+1))
        pygame.draw.rect(surface,out, (cx-2*u,bot-7*u,4*u,u+1),1)
        pygame.draw.line(surface,out,(cx,bot-7*u-1),(cx,bot-10*u),3)
        pygame.draw.line(surface,out,(cx-u,bot-9*u),(cx+u,bot-9*u),3)


STATE_HUMAN    = "human"
STATE_AI       = "ai"
STATE_GAMEOVER = "gameover"


def run_cli():
    board = Board()
    print(board.display())
    while True:
        mg    = MoveGenerator(board)
        legal = mg.generate_legal_moves(board.turn)
        if not legal:
            print("Checkmate!" if board.is_in_check(board.turn) else "Stalemate.")
            break
        if board.is_draw_by_fifty_moves() or board.is_insufficient_material():
            print("Draw."); break
        if board.turn==HUMAN_COLOR:
            while True:
                raw = input("Your move (e.g. e2e4): ").strip().lower()
                m   = next((x for x in legal if str(x)==raw), None)
                if m: board.make_move(m); break
                print("Illegal. Legal:", ", ".join(str(x) for x in legal))
        else:
            print("AI thinking...")
            move = get_ai_move(board)
            if move: print(f"AI: {move}"); board.make_move(move)
        print(board.display()); print()


def run_pygame():
    pygame.init()
    SQ=80; PANEL=200
    W=SQ*8+PANEL; H=SQ*8
    screen = pygame.display.set_mode((W,H))
    pygame.display.set_caption("Chess Engine — Human vs AI")

    try:    ui  = pygame.font.SysFont("Arial",17)
    except: ui  = pygame.font.Font(None,20)
    try:    big = pygame.font.SysFont("Arial",32,bold=True)
    except: big = pygame.font.Font(None,36)

    LIGHT=(240,217,181); DARK=(181,136,99); BG=(35,35,35)
    C_SEL=(100,200,100,150); C_LEG=(70,130,210,120); C_LST=(200,200,50,130)

    board=Board()
    state=STATE_HUMAN if HUMAN_COLOR==WHITE else STATE_AI
    sel=None; legal_dests=[]; last_move=None
    msg="Your turn" if state==STATE_HUMAN else "AI thinking..."
    flip=BOARD_FLIP
    ai_result=[None]; ai_thread=[None]

    def to_scr(r,c):
        return ((7-c)*SQ if flip else c*SQ, r*SQ)

    def from_scr(px,py):
        c=7-px//SQ if flip else px//SQ
        return py//SQ, c

    def legal_for(sq):
        mg=MoveGenerator(board)
        return [m.to_sq for m in mg.generate_legal_moves(HUMAN_COLOR) if m.from_sq==sq]

    def find_move(fsq,tsq):
        mg=MoveGenerator(board)
        cands=[m for m in mg.generate_legal_moves(HUMAN_COLOR) if m.from_sq==fsq and m.to_sq==tsq]
        if not cands: return None
        for m in cands:
            if m.promotion!=EMPTY and abs(m.promotion)==QUEEN: return m
        return cands[0]

    def start_ai():
        b2=board.copy()
        ai_result[0]=None
        def worker():
            mv=get_ai_move(b2)
            if mv is None: ai_result[0]=None; return
            uci=str(mv)
            for m in MoveGenerator(board).generate_legal_moves(board.turn):
                if str(m)==uci: ai_result[0]=m; return
            legal=MoveGenerator(board).generate_legal_moves(board.turn)
            ai_result[0]=legal[0] if legal else None
        t=threading.Thread(target=worker,daemon=True); t.start()
        ai_thread[0]=t

    def draw_board():
        for r in range(8):
            for c in range(8):
                col=LIGHT if (r+c)%2==0 else DARK
                x,y=to_scr(r,c)
                pygame.draw.rect(screen,col,(x,y,SQ,SQ))

    def draw_overlay():
        surf=pygame.Surface((SQ,SQ),pygame.SRCALPHA)
        if last_move:
            for sq in [last_move.from_sq,last_move.to_sq]:
                surf.fill(C_LST); x,y=to_scr(*sq); screen.blit(surf,(x,y))
        if sel:
            surf.fill(C_SEL); x,y=to_scr(*sel); screen.blit(surf,(x,y))
        surf.fill(C_LEG)
        for sq in legal_dests:
            x,y=to_scr(*sq); screen.blit(surf,(x,y))

    def draw_pieces():
        for r in range(8):
            for c in range(8):
                p=int(board.squares[r][c])
                if p==0: continue
                x,y=to_scr(r,c)
                draw_piece(screen,abs(p),p>0,x,y,SQ)

    def draw_panel():
        px=SQ*8
        pygame.draw.rect(screen,BG,(px,0,PANEL,H))
        turn_s="White" if board.turn==WHITE else "Black"
        ai_s  ="Black" if HUMAN_COLOR==WHITE else "White"
        items=[
            (big,"CHESS ENGINE",(210,180,100)),
            (ui, "",            (0,0,0)),
            (ui, f"Turn: {turn_s}",(230,230,230)),
            (ui, f"Move: {max(1,board.fullmove_number)}",(180,180,180)),
            (ui, f"AI: {ai_s}",(160,160,160)),
            (ui, f"Depth: {AI_DEPTH}  Time: {AI_TIME_LIMIT}s",(140,140,140)),
            (ui, "",            (0,0,0)),
            (ui, msg,           (100,255,150)),
            (ui, "",            (0,0,0)),
            (ui, "R: Restart",  (120,120,120)),
            (ui, "F: Flip",     (120,120,120)),
            (ui, "ESC: Quit",   (120,120,120)),
        ]
        y=15
        for font,text,col in items:
            if text: screen.blit(font.render(text,True,col),(px+10,y))
            y+=font.get_height()+5

    if state==STATE_AI: start_ai()
    clock=pygame.time.Clock()
    running=True

    while running:
        screen.fill(BG)
        draw_board(); draw_overlay(); draw_pieces(); draw_panel()
        pygame.display.flip()
        clock.tick(30)

        # AI result check
        if state==STATE_AI:
            t=ai_thread[0]
            if t and not t.is_alive():
                move=ai_result[0]
                if move:
                    if board.make_move(move):
                        last_move=move; msg=f"AI played {move}"
                    else:
                        msg="AI move error"
                else:
                    msg="AI: no move found"
                mg2=MoveGenerator(board)
                l2=mg2.generate_legal_moves(board.turn)
                if not l2:
                    msg=("Checkmate! "+(("Black" if board.turn==WHITE else "White")+" wins.")
                         if board.is_in_check(board.turn) else "Stalemate — draw.")
                    state=STATE_GAMEOVER
                elif board.is_draw_by_fifty_moves() or board.is_insufficient_material():
                    msg="Draw."; state=STATE_GAMEOVER
                else:
                    state=STATE_HUMAN; msg="Your turn"

        for event in pygame.event.get():
            if event.type==pygame.QUIT: running=False
            elif event.type==pygame.KEYDOWN:
                if event.key==pygame.K_ESCAPE: running=False
                elif event.key==pygame.K_r:
                    board=Board()
                    state=STATE_HUMAN if HUMAN_COLOR==WHITE else STATE_AI
                    sel=None; legal_dests=[]; last_move=None
                    msg="Your turn" if state==STATE_HUMAN else "AI thinking..."
                    if state==STATE_AI: start_ai()
                elif event.key==pygame.K_f:
                    flip=not flip
            elif event.type==pygame.MOUSEBUTTONDOWN and state==STATE_HUMAN:
                px,py=event.pos
                if px>=SQ*8: continue
                r,c=from_scr(px,py)
                if not(0<=r<8 and 0<=c<8): continue
                clicked=(r,c)
                p=int(board.squares[r][c])
                if sel is None:
                    if p!=0 and (1 if p>0 else -1)==HUMAN_COLOR:
                        sel=clicked; legal_dests=legal_for(clicked)
                else:
                    if clicked in legal_dests:
                        move=find_move(sel,clicked)
                        if move and board.make_move(move):
                            last_move=move; sel=None; legal_dests=[]
                            mg3=MoveGenerator(board)
                            l3=mg3.generate_legal_moves(board.turn)
                            if not l3:
                                msg=("Checkmate! "+(("Black" if board.turn==WHITE else "White")+" wins.")
                                     if board.is_in_check(board.turn) else "Stalemate — draw.")
                                state=STATE_GAMEOVER
                            elif board.is_draw_by_fifty_moves() or board.is_insufficient_material():
                                msg="Draw."; state=STATE_GAMEOVER
                            else:
                                state=STATE_AI; msg="AI thinking..."; start_ai()
                        else:
                            sel=None; legal_dests=[]
                    elif p!=0 and (1 if p>0 else -1)==HUMAN_COLOR:
                        sel=clicked; legal_dests=legal_for(clicked)
                    else:
                        sel=None; legal_dests=[]

    pygame.quit()


if __name__=="__main__":
    if "--cli" in sys.argv or not PYGAME_AVAILABLE:
        run_cli()
    else:
        run_pygame()