"""
extract_keypoints.py
====================
Run this ONCE on your video folder to generate a labeled CSV for training.

FOLDER STRUCTURE EXPECTED:
datasets/
  raw/
    sitting/        ← put all sitting videos here
    eating/
    walking/
    pill_taking/
    lying_down/
    no_person/

USAGE:
  python3 scripts/extract_keypoints.py

OUTPUT:
  datasets/carewatch_labeled.csv   ← ready for training
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import torch

# ── CONFIG ────────────────────────────────────────────────────────────────────
RAW_VIDEO_DIR   = Path("datasets/raw")
OUTPUT_CSV      = Path("datasets/carewatch_labeled.csv")
POSE_MODEL      = "yolo11x-pose.pt"
FRAME_SKIP      = 3        # process every 3rd frame (speeds things up 3x)
MIN_CONFIDENCE  = 0.5      # ignore keypoints below this confidence
DEVICE          = "mps" if torch.backends.mps.is_available() else "cpu"

LABELS = ["sitting", "eating", "walking", "pill_taking", "lying_down", "no_person", "fallen"]

# YOLO11x-pose gives 17 keypoints in COCO format:
# 0:nose 1:left_eye 2:right_eye 3:left_ear 4:right_ear
# 5:left_shoulder 6:right_shoulder 7:left_elbow 8:right_elbow
# 9:left_wrist 10:right_wrist 11:left_hip 12:right_hip
# 13:left_knee 14:right_knee 15:left_ankle 16:right_ankle
KEYPOINT_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle"
]

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
print(f"Loading YOLO pose model on {DEVICE}...")
model = YOLO(POSE_MODEL)
print("Model loaded.")

# ── HELPERS ───────────────────────────────────────────────────────────────────

def extract_keypoints_from_frame(frame: np.ndarray) -> np.ndarray | None:
    """
    Run YOLO pose on one frame.
    Returns flat array of [x0,y0, x1,y1, ..., x16,y16] for the most confident person.
    Returns None if no person detected or confidence too low.
    """
    results = model.predict(frame, conf=0.25, verbose=False)[0]

    if results.keypoints is None or len(results.keypoints.data) == 0:
        return None

    # If multiple people, take the one with highest average keypoint confidence
    kp_data = results.keypoints.data.cpu().numpy()  # shape: (num_people, 17, 3)

    best_person = None
    best_conf = -1
    for person_kps in kp_data:
        avg_conf = person_kps[:, 2].mean()
        if avg_conf > best_conf:
            best_conf = avg_conf
            best_person = person_kps

    if best_conf < MIN_CONFIDENCE:
        return None

    # Flatten to [x0,y0, x1,y1, ..., x16,y16] — 34 values
    xy = best_person[:, :2].flatten()
    return xy


def process_video(video_path: Path, label: str) -> list[dict]:
    """
    Process one video file. Returns list of row dicts ready for DataFrame.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ⚠️  Could not open {video_path.name}, skipping.")
        return []

    rows = []
    frame_idx = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % FRAME_SKIP != 0:
            continue

        keypoints = extract_keypoints_from_frame(frame)

        if keypoints is None:
            # For no_person label, a missing detection IS the signal
            if label == "no_person":
                row = {"label": label, "frame": frame_idx, "source": video_path.name}
                row.update({f"{name}_x": 0.0 for name in KEYPOINT_NAMES})
                row.update({f"{name}_y": 0.0 for name in KEYPOINT_NAMES})
                rows.append(row)
            continue

        row = {"label": label, "frame": frame_idx, "source": video_path.name}
        for i, name in enumerate(KEYPOINT_NAMES):
            row[f"{name}_x"] = float(keypoints[i*2])
            row[f"{name}_y"] = float(keypoints[i*2 + 1])

        rows.append(row)
        processed += 1

    cap.release()
    print(f"  ✅ {video_path.name}: {processed} frames extracted")
    return rows


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    all_rows = []

    for label in LABELS:
        label_dir = RAW_VIDEO_DIR / label
        if not label_dir.exists():
            print(f"⚠️  Folder not found: {label_dir} — skipping {label}")
            continue

        video_files = list(label_dir.glob("*.mp4")) + \
                      list(label_dir.glob("*.MOV")) + \
                      list(label_dir.glob("*.mov")) + \
                      list(label_dir.glob("*.MP4"))

        if not video_files:
            print(f"⚠️  No videos found in {label_dir} — skipping {label}")
            continue

        print(f"\n📁 Processing label: {label} ({len(video_files)} videos)")
        for vf in video_files:
            rows = process_video(vf, label)
            all_rows.extend(rows)

    if not all_rows:
        print("\n❌ No data extracted. Check your folder structure.")
        return

    df = pd.DataFrame(all_rows)

    # Shuffle so labels aren't in blocks
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Train/test split (80/20)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df  = df.iloc[split_idx:]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv("datasets/train_action_pose_keypoint.csv", index=False)
    test_df.to_csv("datasets/test_action_pose_keypoint.csv",   index=False)

    print(f"\n✅ Done!")
    print(f"   Total frames : {len(df)}")
    print(f"   Training rows: {len(train_df)}")
    print(f"   Test rows    : {len(test_df)}")
    print(f"\n   Label distribution:")
    print(df['label'].value_counts().to_string())
    print(f"\n   Saved to datasets/train_action_pose_keypoint.csv")
    print(f"           datasets/test_action_pose_keypoint.csv")


if __name__ == "__main__":
    main()