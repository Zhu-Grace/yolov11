from collections import defaultdict

import cv2
import numpy
import torch
from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO(model="yolo11n.pt",task='detect')
  
# Open the video file
video_path = "D:\zhujiaxuan\project\yolov11\ultralytics\trackers\tennis.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        print(type(frame))
        print(frame.shape)
        # Check if frame is a numpy array
        # if isinstance(frame, numpy.ndarray):
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        # frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        # frame_tensor = frame_tensor.unsqueeze(0)  # Add batch dimension
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(numpy.uint8)
        frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

        # Add a batch dimension
        # frame = frame_tensor.unsqueeze(0)


        results = model.track(frame, persist=True)
        # results = model.track(frame_tensor, persist=True)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 30 tracks for 30 frames
                track.pop(0)

            # Draw the tracking lines
            points = numpy.hstack(track).astype(numpy.int32).reshape((-1, 1, 2))
            cv2.polylines(
                annotated_frame,
                [points],
                isClosed=False,
                color=(230, 230, 230),
                thickness=10,
            )

        # Display the annotated frame
        cv2.imshow("YOLO11 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        print("Frame is not a numpy array.")
        break
    # else:
    #     # Break the loop if the end of the video is reached
    #     break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
