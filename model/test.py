from ultralytics import YOLO

# Automatically downloads and caches the model if not already present
detection_model = YOLO('yolo11n.pt')
pose_model = YOLO('yolo11x-pose.pt')
