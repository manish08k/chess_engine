"""
ml_model.py - Machine Learning Position Evaluation

Uses a RandomForestRegressor to learn position evaluation from
self-generated training data (positions evaluated by the classical
engine). The ML model can then provide fast evaluations as a
complement to or replacement for the hand-crafted evaluation function.

Training flow:
1. Generate random positions via self-play/perturbation
2. Evaluate each position with the classical evaluator
3. Extract board features into a feature vector
4. Train RandomForestRegressor on (features, score) pairs
5. Use trained model for fast inference during search
"""

import os
import pickle
import numpy as np
from typing import Optional, List, Tuple
from board import Board, Move, WHITE, BLACK, EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING
from evaluation import Evaluator, PIECE_VALUES

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[MLModel] Warning: scikit-learn not available. ML evaluation disabled.")


# ------------------------------------------------------------------
# Feature Extraction
# ------------------------------------------------------------------

def extract_features(board: Board) -> np.ndarray:
    """
    Extract a fixed-size feature vector from a board position.

    Features include:
    - Material balance for each piece type (12 features)
    - Piece counts per side (12 features)
    - Pawn structure metrics (8 features)
    - King safety metrics (4 features)
    - Mobility estimates (4 features)
    - Game phase indicators (2 features)
    - Center control (4 features)
    Total: 46 features
    """
    features = []

    # --- Material counts (piece counts per color) ---
    piece_counts = {
        WHITE: {PAWN: 0, KNIGHT: 0, BISHOP: 0, ROOK: 0, QUEEN: 0},
        BLACK: {PAWN: 0, KNIGHT: 0, BISHOP: 0, ROOK: 0, QUEEN: 0},
    }

    for r in range(8):
        for c in range(8):
            p = int(board.squares[r][c])
            if p == EMPTY:
                continue
            color = WHITE if p > 0 else BLACK
            abs_p = abs(p)
            if abs_p in piece_counts[color]:
                piece_counts[color][abs_p] += 1

    # Feature: material balance per piece type
    for pt in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN]:
        diff = piece_counts[WHITE][pt] - piece_counts[BLACK][pt]
        features.append(diff)  # 5 features

    # Feature: raw piece counts
    for color in [WHITE, BLACK]:
        for pt in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN]:
            features.append(piece_counts[color][pt])  # 10 features

    # --- Material balance (total) ---
    white_material = sum(piece_counts[WHITE][pt] * PIECE_VALUES[pt] for pt in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN])
    black_material = sum(piece_counts[BLACK][pt] * PIECE_VALUES[pt] for pt in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN])
    features.append(white_material - black_material)  # 1 feature
    features.append(white_material + black_material)  # Total material (game phase)

    # --- Pawn structure ---
    # Count doubled, isolated, and passed pawns per side
    for color in [WHITE, BLACK]:
        pawn = color * PAWN
        pawn_files = []
        pawn_positions = []

        for r in range(8):
            for c in range(8):
                if board.squares[r][c] == pawn:
                    pawn_files.append(c)
                    pawn_positions.append((r, c))

        # Doubled pawns
        doubled = sum(pawn_files.count(f) - 1 for f in set(pawn_files) if pawn_files.count(f) > 1)
        features.append(doubled)  # 2 features (one per color)

        # Isolated pawns
        isolated = sum(1 for f in pawn_files if (f-1) not in pawn_files and (f+1) not in pawn_files)
        features.append(isolated)  # 2 features

        # Pawn advancement (sum of how far pawns have advanced)
        if color == WHITE:
            advancement = sum(6 - r for r, c in pawn_positions)
        else:
            advancement = sum(r - 1 for r, c in pawn_positions)
        features.append(advancement)  # 2 features

    # --- King safety ---
    for color in [WHITE, BLACK]:
        king_pos = board.king_positions.get(color)
        if king_pos:
            kr, kc = king_pos
            # King centralization (endgame: center is good, middlegame: corner is good)
            center_dist = abs(kr - 3.5) + abs(kc - 3.5)
            features.append(center_dist)  # 2 features

            # Pawn shield count
            direction = -1 if color == WHITE else 1
            shield_row = kr + direction
            shield = 0
            if 0 <= shield_row < 8:
                for dc in range(max(0, kc - 1), min(8, kc + 2)):
                    if board.squares[shield_row][dc] == color * PAWN:
                        shield += 1
            features.append(shield)  # 2 features
        else:
            features.extend([0, 0])  # Safety fallback

    # --- Mobility (approximate: count occupied squares adjacent to own pieces) ---
    for color in [WHITE, BLACK]:
        mobility = 0
        for r in range(8):
            for c in range(8):
                p = int(board.squares[r][c])
                if p == EMPTY or (WHITE if p > 0 else BLACK) != color:
                    continue
                # Count empty squares around piece (simplified mobility)
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < 8 and 0 <= nc < 8 and board.squares[nr][nc] == EMPTY:
                            mobility += 1
        features.append(mobility)  # 2 features

    # --- Center control ---
    # Pieces and pawns on center squares
    center_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]
    extended_center = [(2, 2), (2, 3), (2, 4), (2, 5),
                        (3, 2), (3, 5), (4, 2), (4, 5),
                        (5, 2), (5, 3), (5, 4), (5, 5)]

    for color in [WHITE, BLACK]:
        center_ctrl = sum(
            1 for r, c in center_squares
            if board.squares[r][c] != EMPTY and (WHITE if int(board.squares[r][c]) > 0 else BLACK) == color
        )
        features.append(center_ctrl)  # 2 features

    # --- Castling rights ---
    from board import CASTLE_WK, CASTLE_WQ, CASTLE_BK, CASTLE_BQ
    features.append(1 if board.castling_rights & (CASTLE_WK | CASTLE_WQ) else 0)
    features.append(1 if board.castling_rights & (CASTLE_BK | CASTLE_BQ) else 0)

    # --- Turn ---
    features.append(1 if board.turn == WHITE else -1)

    return np.array(features, dtype=np.float32)


# ------------------------------------------------------------------
# ML Model
# ------------------------------------------------------------------

class MLEvaluator:
    """
    Machine learning position evaluator using RandomForestRegressor.

    Trained on positions evaluated by the classical Evaluator.
    Provides fast approximate evaluation as an alternative to the
    full classical evaluator.
    """

    MODEL_PATH = "data/ml_model.pkl"

    def __init__(self):
        self.model: Optional[RandomForestRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self.classical = Evaluator()
        self.trained = False

        if SKLEARN_AVAILABLE:
            self._load_or_train()

    def _load_or_train(self):
        """Load a pre-trained model or train a new one."""
        if os.path.exists(self.MODEL_PATH):
            try:
                with open(self.MODEL_PATH, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                    self.scaler = data['scaler']
                    self.trained = True
                print(f"[MLEvaluator] Loaded model from {self.MODEL_PATH}")
                return
            except Exception as e:
                print(f"[MLEvaluator] Error loading model: {e}")

        print("[MLEvaluator] Training new model...")
        self.train()

    def train(self, n_positions: int = 1000):
        """
        Generate training data and train the RandomForestRegressor.

        Args:
            n_positions: Number of positions to generate for training
        """
        if not SKLEARN_AVAILABLE:
            print("[MLEvaluator] scikit-learn not available, skipping training")
            return

        print(f"[MLEvaluator] Generating {n_positions} training positions...")
        X, y = self._generate_training_data(n_positions)

        if len(X) < 50:
            print("[MLEvaluator] Insufficient training data")
            return

        X_arr = np.array(X)
        y_arr = np.array(y)

        # Split into train/validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_arr, y_arr, test_size=0.2, random_state=42
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Train RandomForestRegressor
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            n_jobs=-1,  # Use all CPU cores
            random_state=42,
        )
        self.model.fit(X_train_scaled, y_train)

        # Evaluate on validation set
        val_pred = self.model.predict(X_val_scaled)
        mae = mean_absolute_error(y_val, val_pred)
        print(f"[MLEvaluator] Training complete. Validation MAE: {mae:.1f} cp")

        self.trained = True
        self._save()

    def _generate_training_data(self, n: int) -> Tuple[List[np.ndarray], List[float]]:
        """
        Generate training positions by playing random games and
        evaluating each position with the classical evaluator.
        """
        from move_generator import MoveGenerator
        import random

        X, y = [], []
        board = Board()

        for game_idx in range(max(1, n // 30)):
            board = Board()

            for move_num in range(random.randint(5, 40)):
                mg = MoveGenerator(board)
                moves = mg.generate_legal_moves(board.turn)
                if not moves:
                    break

                # Mostly play reasonable moves, sometimes random
                if random.random() < 0.7:
                    # MVV-LVA ordered move
                    moves = mg.order_moves(moves)
                    move = moves[min(random.randint(0, 2), len(moves) - 1)]
                else:
                    move = random.choice(moves)

                board.make_move(move)

                # Sample this position
                features = extract_features(board)
                score = self.classical.evaluate(board)
                X.append(features)
                y.append(float(score))

                if len(X) >= n:
                    return X, y

        return X, y

    def _save(self):
        """Save trained model to disk."""
        os.makedirs(os.path.dirname(self.MODEL_PATH) if os.path.dirname(self.MODEL_PATH) else '.', exist_ok=True)
        with open(self.MODEL_PATH, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
        print(f"[MLEvaluator] Model saved to {self.MODEL_PATH}")

    def evaluate(self, board: Board) -> int:
        """
        Evaluate position using the ML model.
        Returns score in centipawns (White's perspective).
        Falls back to classical evaluation if model not trained.
        """
        if not self.trained or self.model is None or self.scaler is None:
            return self.classical.evaluate(board)

        try:
            features = extract_features(board).reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            score = self.model.predict(features_scaled)[0]
            return int(score)
        except Exception:
            return self.classical.evaluate(board)

    def evaluate_for_side(self, board: Board, color: int) -> int:
        """Return evaluation from the perspective of the given color."""
        score = self.evaluate(board)
        return score if color == WHITE else -score

    @property
    def is_available(self) -> bool:
        return SKLEARN_AVAILABLE and self.trained
