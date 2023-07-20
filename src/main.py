import os
import cv2
from ultralytics import YOLO
from tracker import Tracker
import random

video_path = os.path.join('.', 'data','people.mp4')
video_out_path = os.path.join('.', 'output.mp4')

# Capture the video
cap = cv2.VideoCapture(video_path)

ret,frame = cap.read()

# load the model
model = YOLO("yolov8n.pt")

# save the object detected video
cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))

# load the tracker
tracker = Tracker()

# create 10 colors for rectangle boxes
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

while ret:
    results = model(frame)
  
    # unwrap the tensors
    for result in results:
        detection = []
        for r in result.boxes.data.tolist():
            x1,y1,x2,y2,score,class_id = r
            detection.append([x1,y1,x2,y2,score])

        # call the deep_sort tracker
        tracker.update(frame=frame, detections=detection)

        # track objects
        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id

            # draw rectangles on detected objects
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)

    # write the frame
    cap_out.write(frame)

    # read new frame
    ret,frame = cap.read()

cap.release()
cap_out.release()
cv2.destroyAllWindows()