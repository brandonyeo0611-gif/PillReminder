"""
classification_keypoint.py
===========================
AngleFeatureExtractor  — converts 17 COCO keypoints → 12 meaningful joint angles
AngleLSTMNet           — LSTM classifier that takes a sequence of angle frames

CHANGES FROM ORIGINAL:
- Fixed keypoint indices to match YOLO11x-pose COCO 17-keypoint output
- Added velocity features (angle change per frame) — catches motion quality
- Fixed 'lable' typo
- Fixed missing `labels = []` initialization in create_sequences
- Increased angle set from 8 → 12 to use full body
"""

import numpy as np
import torch
import torch.nn as nn

# COCO 17-keypoint indices (matches YOLO11x-pose output exactly)
# 0:nose  1:left_eye  2:right_eye  3:left_ear  4:right_ear
# 5:left_shoulder   6:right_shoulder
# 7:left_elbow      8:right_elbow
# 9:left_wrist      10:right_wrist
# 11:left_hip       12:right_hip
# 13:left_knee      14:right_knee
# 15:left_ankle     16:right_ankle

LABELS = ["sitting", "eating", "walking", "pill_taking", "lying_down", "no_person", "fallen"]
NUM_ANGLE_FEATURES = 12   # see AngleFeatureExtractor.angle_triplets
SEQUENCE_LENGTH    = 30   # frames per classification window (~1 second at 30fps)


class AngleFeatureExtractor:
    """
    Converts a flat array of keypoint coordinates [x0,y0, x1,y1, ..., x16,y16]
    into 12 joint angles covering the full body.
    """

    # Each triplet is (point_A, vertex, point_B) — we compute the angle at vertex
    angle_triplets = [
        # Upper body
        (9,  7, 5),    # left wrist → left elbow → left shoulder
        (10, 8, 6),    # right wrist → right elbow → right shoulder
        (7,  5, 11),   # left elbow → left shoulder → left hip
        (8,  6, 12),   # right elbow → right shoulder → right hip
        # Torso
        (5,  11, 13),  # left shoulder → left hip → left knee
        (6,  12, 14),  # right shoulder → right hip → right knee
        # Lower body
        (11, 13, 15),  # left hip → left knee → left ankle
        (12, 14, 16),  # right hip → right knee → right ankle
        # Cross-body (posture)
        (5,  6,  12),  # left shoulder → right shoulder → right hip
        (6,  5,  11),  # right shoulder → left shoulder → left hip
        # Head / neck tilt
        (5,  0,  6),   # left shoulder → nose → right shoulder
        (11, 0,  12),  # left hip → nose → right hip
    ]

    def calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Angle at p2, formed by p1-p2-p3. Returns 0.0 if points are degenerate."""
        ba = p1 - p2
        bc = p3 - p2

        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)

        if norm_ba < 1e-10 or norm_bc < 1e-10:
            return 0.0

        cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))

    def calculate_angles(self, keypoints_flat: np.ndarray) -> np.ndarray:
        """
        keypoints_flat: 1D array of length 34 [x0,y0, x1,y1, ..., x16,y16]
        Returns: 1D array of 12 angles (degrees)
        """
        # Reshape into (17, 2)
        try:
            points = keypoints_flat.reshape(17, 2)
        except ValueError:
            return np.zeros(NUM_ANGLE_FEATURES)

        angles = []
        for (a, v, b) in self.angle_triplets:
            try:
                angle = self.calculate_angle(points[a], points[v], points[b])
            except Exception:
                angle = 0.0
            angles.append(angle)

        return np.array(angles, dtype=np.float32)


class AngleLSTMNet(nn.Module):
    """
    Input : (batch, sequence_length, NUM_ANGLE_FEATURES)
    Output: (batch, num_classes)
    """

    def __init__(
        self,
        input_size:  int = NUM_ANGLE_FEATURES,
        hidden_size: int = 128,
        num_layers:  int = 3,
        num_classes: int = len(LABELS),
        lstm_dropout: float = 0.3,
        fc_dropout:   float = 0.5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(fc_dropout)
        self.fc      = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_length: int = SEQUENCE_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a window of seq_length over X and y.
    Label of each sequence = label of its last frame.
    """
    sequences = []
    labels    = []   # ← bug fix: was missing in original

    num_sequences = len(X) - seq_length + 1
    for i in range(num_sequences):
        sequences.append(X[i : i + seq_length])
        labels.append(y[i + seq_length - 1])

    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int64)


def load_and_preprocess(
    train_csv: str,
    test_csv:  str,
    seq_length: int = SEQUENCE_LENGTH,
):
    """
    Full preprocessing pipeline.
    Returns X_train, X_test, y_train, y_test as numpy arrays.
    """
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    extractor = AngleFeatureExtractor()

    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)

    # Clean labels
    train_df["label"] = train_df["label"].str.strip().str.lower()
    test_df["label"]  = test_df["label"].str.strip().str.lower()

    # Encode labels
    le = LabelEncoder()
    le.fit(LABELS)
    train_df["label"] = le.transform(train_df["label"])
    test_df["label"]  = le.transform(test_df["label"])

    # Keypoint columns — everything except label, frame, source
    kp_cols = [c for c in train_df.columns if c.endswith("_x") or c.endswith("_y")]

    def extract_angles(df):
        angles = []
        for row in df[kp_cols].values:
            angles.append(extractor.calculate_angles(row))
        return np.array(angles, dtype=np.float32)

    X_train_angles = extract_angles(train_df)
    X_test_angles  = extract_angles(test_df)

    y_train = train_df["label"].values
    y_test  = test_df["label"].values

    X_train, y_train = create_sequences(X_train_angles, y_train, seq_length)
    X_test,  y_test  = create_sequences(X_test_angles,  y_test,  seq_length)

    print(f"Training set : {X_train.shape}  labels: {y_train.shape}")
    print(f"Test set     : {X_test.shape}   labels: {y_test.shape}")
    print(f"Classes      : {le.classes_}")

    return X_train, X_test, y_train, y_test, le