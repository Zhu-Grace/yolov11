from ultralytics import YOLO

# Load an official or custom model
model = YOLO("D:\zhujiaxuan\project\yolov11\models\yolo11n.pt")  # Load an official Detect model
# model = YOLO("yolo11n-seg.pt")  # Load an official Segment model
# model = YOLO("yolo11n-pose.pt")  # Load an official Pose model
# model = YOLO("path/to/best.pt")  # Load a custom trained model

# Perform tracking with the model
results = model(source="test.mp4", show=True)  # Tracking with default tracker
# results = model.track(
#     source="https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml"
# )  # Tracking with ByteTrack tracker