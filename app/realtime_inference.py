"""
realtime_inference.py
======================
The LIVE DEMO script. Opens webcam (or video file), runs the full CareWatch pipeline:
  YOLO pose → angle extraction → LSTM classification → live overlay

USAGE:
  # Webcam
  python3 app/realtime_inference.py

  # Video file (for testing without webcam)
  python3 app/realtime_inference.py --source path/to/video.mp4

CONTROLS:
  Q — quit
  S — save current frame as screenshot
"""

import sys, os, argparse, time, collections, threading
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.classification_keypoint import AngleFeatureExtractor, AngleLSTMNet, SEQUENCE_LENGTH, NUM_ANGLE_FEATURES
from src.logger import ActivityLogger
from src.deviation_detector import DeviationDetector
from src.alert_system import AlertSystem
from src.medication import MedicationRepo
from src.tts import speak

# ── CONFIG ────────────────────────────────────────────────────────────────────
POSE_MODEL_PATH  = "yolo11x-pose.pt"
LSTM_MODEL_PATH  = "model/trained_carewatch.pt"
LABEL_CLASS_PATH = "model/label_classes.txt"
CONFIDENCE_THRESHOLD = 0.60   # minimum LSTM confidence to display prediction
MIN_KEYPOINT_CONF    = 0.40

# Activity colours (BGR)
ACTIVITY_COLOURS = {
    "sitting":    (180, 180, 180),
    "eating":     (0,   200, 100),
    "walking":    (0,   160, 255),
    "pill_taking":(0,   255, 255),
    "pill_intake":(0,   255, 255),
    "lying_down": (200, 100, 200),
    "no_person":  (80,   80,  80),
    "fallen":     (0,    0,  255),  # red for urgent
    "unknown":    (100, 100, 100),
}

ACTIVITY_ALIASES = {
    "pill_intake": "pill_taking",
}

# ── LOAD MODELS ───────────────────────────────────────────────────────────────

def load_models():
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pose_model = YOLO(POSE_MODEL_PATH)

    # Load label classes
    if not os.path.exists(LABEL_CLASS_PATH):
        print(f"⚠️  {LABEL_CLASS_PATH} not found. Run training first.")
        label_classes = ["sitting","eating","walking","pill_taking","lying_down","no_person","fallen"]
    else:
        with open(LABEL_CLASS_PATH) as f:
            label_classes = [l.strip() for l in f.readlines()]

    num_classes = len(label_classes)

    # Load LSTM
    lstm_model = AngleLSTMNet(
        input_size=NUM_ANGLE_FEATURES,
        hidden_size=128,
        num_layers=3,
        num_classes=num_classes,
    ).to(device)

    if os.path.exists(LSTM_MODEL_PATH):
        lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=device))
        lstm_model.eval()
        print(f"✅ LSTM model loaded from {LSTM_MODEL_PATH}")
    else:
        print(f"⚠️  No trained model found at {LSTM_MODEL_PATH}. Running pose-only mode.")
        lstm_model = None

    return pose_model, lstm_model, label_classes, device


# ── KEYPOINT EXTRACTION ───────────────────────────────────────────────────────

def extract_keypoints(results) -> np.ndarray | None:
    """Extract flat [x0,y0,...,x16,y16] for most confident person."""
    if results.keypoints is None or len(results.keypoints.data) == 0:
        return None

    kp_data = results.keypoints.data.cpu().numpy()
    best, best_conf = None, -1
    for person in kp_data:
        conf = person[:, 2].mean()
        if conf > best_conf:
            best_conf, best = conf, person
    if best_conf < MIN_KEYPOINT_CONF:
        return None

    return best[:, :2].flatten()   # (34,)


# ── DRAWING HELPERS ───────────────────────────────────────────────────────────

SKELETON_CONNECTIONS = [
    (5, 7), (7, 9),    # left arm
    (6, 8), (8, 10),   # right arm
    (5, 6),            # shoulders
    (5, 11), (6, 12),  # torso sides
    (11, 12),          # hips
    (11, 13), (13, 15),# left leg
    (12, 14), (14, 16),# right leg
    (0, 5), (0, 6),    # neck to shoulders
]

def draw_skeleton(frame, keypoints_flat, colour=(0, 255, 255)):
    if keypoints_flat is None:
        return
    pts = keypoints_flat.reshape(17, 2).astype(int)
    for (a, b) in SKELETON_CONNECTIONS:
        pa, pb = tuple(pts[a]), tuple(pts[b])
        if pa != (0,0) and pb != (0,0):
            cv2.line(frame, pa, pb, colour, 2)
    for pt in pts:
        if tuple(pt) != (0,0):
            cv2.circle(frame, tuple(pt), 4, (255, 255, 255), -1)


def draw_overlay(frame, activity, confidence, fps):
    h, w = frame.shape[:2]
    colour = ACTIVITY_COLOURS.get(activity, ACTIVITY_COLOURS["unknown"])

    # Activity pill (top-left)
    label_text = f"{activity.upper().replace('_', ' ')}  {confidence*100:.0f}%"
    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
    cv2.rectangle(frame, (10, 10), (tw + 30, th + 30), colour, -1)
    cv2.putText(frame, label_text, (20, th + 18),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 2)

    # FPS (top-right)
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 120, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # CareWatch watermark (bottom-left)
    cv2.putText(frame, "CareWatch v1.0", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)


# ── DEVIATION CHECK (runs every 15 min, sends Telegram on YELLOW/RED) ──────────
PERSON_ID = "resident_001"  # pseudonymous ID — real name stored separately

def deviation_check_loop(med_repo, logger):
    """Background thread: compare today vs baseline, send alerts if needed, and check TTS reminders."""
    iterations = 0
    while True:
        try:
            # Check deviation every 15 mins (15 iterations of 60s)
            if iterations % 15 == 0:
                result = DeviationDetector().check(PERSON_ID)
                AlertSystem().send(result, person_name="Resident")  # display name, not stored

            # Check for missed meds and meals every minute
            med_repo.check_and_trigger_reminders(PERSON_ID, speaker=speak)
            med_repo.check_and_trigger_meal_reminders(PERSON_ID, speaker=speak, logger=logger)
            med_repo.check_meal_relative_reminders(PERSON_ID, speaker=speak)
        except Exception as e:
            print(f"⚠️ Background loop warning: {e}")

        iterations += 1
        time.sleep(60)  # 1 minute

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────

def run(source=0):
    pose_model, lstm_model, label_classes, device = load_models()
    extractor = AngleFeatureExtractor()
    logger = ActivityLogger()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Cannot open source: {source}")
        return

    med_repo = MedicationRepo()
    threading.Thread(target=deviation_check_loop, args=(med_repo, logger), daemon=True).start()

    # Rolling buffer of angle frames for LSTM
    frame_buffer = collections.deque(maxlen=SEQUENCE_LENGTH)

    current_activity   = "unknown"
    current_confidence = 0.0
    prev_time          = time.time()
    last_live_update   = 0.0
    last_medication_event_at = {}

    print("▶  CareWatch running. Press Q to quit, S to screenshot.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Pose detection ──
        results = pose_model.predict(frame, conf=0.25, verbose=False)[0]
        keypoints_flat = extract_keypoints(results)

        if keypoints_flat is not None:
            draw_skeleton(frame, keypoints_flat,
                          colour=ACTIVITY_COLOURS.get(current_activity, (0,255,255)))
            angles = extractor.calculate_angles(keypoints_flat)
            frame_buffer.append(angles)
        else:
            frame_buffer.append(np.zeros(NUM_ANGLE_FEATURES, dtype=np.float32))

        # ── LSTM classification (once buffer is full) ──
        if lstm_model is not None and len(frame_buffer) == SEQUENCE_LENGTH:
            seq = torch.tensor(
                np.array(frame_buffer)[np.newaxis, :, :],  # (1, seq, features)
                dtype=torch.float32, device=device
            )
            with torch.no_grad():
                logits = lstm_model(seq)
                probs  = torch.softmax(logits, dim=1)[0]
                top_idx  = probs.argmax().item()
                top_conf = probs[top_idx].item()

            if top_conf >= CONFIDENCE_THRESHOLD:
                predicted_activity = label_classes[top_idx]
                current_activity = ACTIVITY_ALIASES.get(predicted_activity, predicted_activity)
                current_confidence = top_conf
                if current_confidence >= 0.85:
                    logger.log(current_activity, current_confidence)

                    if current_activity == "pill_taking":
                        try:
                            now_dt = datetime.utcnow()
                            schedules = med_repo.list_schedules("resident")
                            nearest = None
                            nearest_delta = None
                            for schedule in schedules:
                                hh, mm = str(schedule.get("time_of_day", "00:00")).split(":")
                                sched_dt = now_dt.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
                                delta_minutes = abs((now_dt - sched_dt).total_seconds()) / 60.0
                                window = max(15, int(schedule.get("tolerance_min") or 30))
                                if delta_minutes <= window:
                                    if nearest is None or delta_minutes < nearest_delta:
                                        nearest = schedule
                                        nearest_delta = delta_minutes

                            if nearest is not None:
                                med_name = nearest.get("medication_name")
                                last_ts = last_medication_event_at.get(med_name)
                                if not last_ts or (now_dt - last_ts).total_seconds() >= 300:
                                    med_repo.record_event("resident", med_name, now_dt, source="ai")
                                    last_medication_event_at[med_name] = now_dt
                                    print(f"✅ medication_event logged: {med_name}")
                        except Exception as e:
                            print(f"⚠️ medication_event logging skipped: {e}")

                    # --- pattern check: if pill_taking detected, confirm it only if
                    # recent logs within a timeframe also include eating and drinking_water.
                    try:
                        if current_activity == "pill_taking":
                            recent = logger.get_recent_minutes(10)
                            acts = {r['activity'] for r in recent}
                            if 'eating' in acts and 'drinking_water' in acts:
                                # Log a derived confirmed-pill event
                                logger.log('pill_intake_confirmed', current_confidence)
                                print('✅ pill_intake_confirmed logged (eating+drinking_water detected)')
                    except Exception:
                        # Don't break realtime loop on logging errors
                        pass
            else:
                current_activity   = "unknown"
                current_confidence = top_conf

        # ── FPS ──
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-9)
        prev_time = now

        # ── Draw overlay ──
        draw_overlay(frame, current_activity, current_confidence, fps)

        # ── Update live status (throttled to once per second) ──
        try:
            now_t = time.time()
            if now_t - last_live_update >= 1.0:
                logger.update_live_status(current_activity, current_confidence)
                last_live_update = now_t
        except Exception:
            pass

        cv2.imshow("CareWatch", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            fname = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, frame)
            print(f"📸 Saved {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print("👋 CareWatch stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=0,
                        help="Camera index (0) or path to video file")
    args = parser.parse_args()

    source = int(args.source) if str(args.source).isdigit() else args.source
    run(source)