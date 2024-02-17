import cv2
import numpy as np
import argparse
from tracker import Tracker
import time


# Load YOLO model
net = cv2.dnn.readNetFromDarknet("yolo_files/yolov3.cfg", "yolo_files/yolov3.weights")
yolo_layers = ['yolo_82', 'yolo_94', 'yolo_106']

def load_labels(labels_path):
    with open(labels_path, 'r') as f:
        return [line.strip() for line in f]

def detect_objects(frame):
    blob = cv2.dnn.blobFromImage(frame, 1/256, (832, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(yolo_layers)
    return outputs

def get_boxes(outputs, height, width, matched_classes):
    boxes_xywh = []  # For processing
    boxes_x1y1x2y2 = []  # For drawing
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if class_id in matched_classes:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes_xywh.append([x, y, w, h])
                boxes_x1y1x2y2.append([x, y, x + w, y+h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes_xywh, boxes_x1y1x2y2, confidences

def yolo_counter(labels, video_path, output_path):
    classes = load_labels(labels)
    car_class_id = classes.index('car')
    bus_class_id = classes.index('bus')
    truck_class_id = classes.index('truck')
    matched_classes = [car_class_id, bus_class_id, truck_class_id]
    cap = cv2.VideoCapture(video_path)
    tracker = Tracker()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    frame_limit = 2000
    frame_interval = 50
    frame_count = 0
    start_time = time.time()

    while cap.isOpened() and frame_count < frame_limit:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        outputs = detect_objects(frame)
        boxes_xywh, boxes_x1y1x2y2, confidences = get_boxes(outputs, height, width, matched_classes)
        indexes = cv2.dnn.NMSBoxes(boxes_xywh, confidences, 0.5, 0.3)
        detections = [boxes_x1y1x2y2[i] for i in indexes]
        centroids = [[int((x+w)/2), int((y+h)/2)] for x, y, w, h in detections]
        tracker.update(centroids)

        # Draw bounding boxes
        for (x1, y1, x2, y2) in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        for objectID, centroid in tracker.objects.items():
            text = f"ID {objectID}"
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            # Draw the path
            for i in range(1, len(tracker.paths[objectID])):
                if len(tracker.paths[objectID]) < 2:
                    break
                thickness = int((64 / float(i + 1)))
                thickness = 10 - thickness
                thickness = max(1, thickness) 
                thickness = min(thickness, 10)
                cv2.line(frame, (tracker.paths[objectID][i - 1][0], tracker.paths[objectID][i - 1][1]),
                         (tracker.paths[objectID][i][0], tracker.paths[objectID][i][1]), (0, 255, 0), 2)

        if out is None:
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

        out.write(frame)

        frame_count += 1
        if frame_count % frame_interval == 0:
            print(f"Current Frame: {frame_count}, Total Elapsed Time: {time.time() - start_time}")


    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("\n------------------------")
    print(f"Total time: {time.time() - start_time}")



def main():
    yolo_counter('./yolo_files/coco.names', 'top_sample.mp4', 'output.mp4')

if __name__ == '__main__':
    main()
