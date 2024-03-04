import cv2
import numpy as np
import argparse
from tracker import Tracker
import time
from lxml import etree
from shapely.geometry import Point, Polygon
from core_logic import *

class Vehicle:
    def __init__(self, vehicle_type, centroid_location, ID, intersection_number, lane_location, action):
        self.vehicle_type = vehicle_type  # "normal_vehicle" or "emergency_vehicle"
        self.centroid_location = centroid_location  # (x, y) of the vehicle's centroid
        self.ID = ID 
        self.intersection_number = intersection_number  # "1", "2", "3", "4"
        self.lane_location = lane_location  # "left", "middle", "right"
        self.action = action  # "stop", "yield_left", "yield_right", "continue"

def parse_anno_file(cvat_xml):
    root = etree.parse(cvat_xml).getroot()
    anno = []
    for image in root.iter('image'):
        lanes = {}
        for polygon in image.iter("polygon"):
            polypoints = polygon.attrib["points"].split(";")
            for i in range(len(polypoints)):
                polypoints[i] = polypoints[i].split(",")
                polypoints[i][0] = float(polypoints[i][0])
                polypoints[i][1] = float(polypoints[i][1])
            lanes[polygon.attrib["label"]] = polypoints
        anno.append(lanes)
    return anno

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

    return boxes_xywh, boxes_x1y1x2y2, confidences, class_ids

def yolo_counter(labels, video_path, mask_path, output_path):
    classes = load_labels(labels)
    car_class_id = classes.index('car')
    bus_class_id = classes.index('bus')
    truck_class_id = classes.index('truck')
    matched_classes = [car_class_id, bus_class_id, truck_class_id]
    cap = cv2.VideoCapture(video_path)
    tracker = Tracker()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    intersection_annotation = parse_anno_file(mask_path)
    starting_frame = 4000
    frame_limit = 4500
    frame_interval = 50
    frame_count = 0
    start_time = time.time()

    vehicle_list = {}
    prev_vehicle_list = {}

    while cap.isOpened() and frame_count < frame_limit:
        ret, frame = cap.read()
        if frame_count < starting_frame:
            frame_count += 1
            continue
        if not ret:
            break

        height, width, _ = frame.shape
        outputs = detect_objects(frame)
        boxes_xywh, boxes_x1y1x2y2, confidences, class_ids = get_boxes(outputs, height, width, matched_classes)
        indexes = cv2.dnn.NMSBoxes(boxes_xywh, confidences, 0.5, 0.3)
        detections = [boxes_x1y1x2y2[i] for i in indexes]
        centroids = [[int((x+w)/2), int((y+h)/2)] for x, y, w, h in detections]
        tracker.update(centroids)

        # Draw bounding boxes
        for i in indexes.flatten():
            (x1, y1, x2, y2) = boxes_x1y1x2y2[i]
            class_id = class_ids[i]  # Access class_id using index i
    
            if class_id in [car_class_id, truck_class_id]:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            elif class_id in [bus_class_id]:  
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255,0), 2)

        for objectID, centroid in tracker.objects.items():
            # Define Vehicle class variable
            if class_ids[objectID] in [car_class_id,truck_class_id]:
                vehicle_type = "normal_vehicle"
            else:
                vehicle_type = "emergency_vehicle"
            centroid_location = centroid
            id = objectID
            intersection_number = " "
            lane_location = " "
            action = " "
            x,y = centroid

            # Conduct lane location detection
            centroid_point = Point(x,y+10)
            for lane, lane_mask in intersection_annotation[frame_count].items():
                lane_polygon = Polygon(lane_mask)
                if centroid_point.within(lane_polygon):
                    splitted_lane = lane.split("_")
                    intersection_number = splitted_lane[0]
                    lane_location = splitted_lane[1]
                    break  # lane annotation should have no overlap
            
            # when vehicles enter the intersection, their lane locations are still retained
            if intersection_number == " " and lane_location == " " and objectID in prev_vehicle_list:
                intersection_number = prev_vehicle_list[objectID].intersection_number
                lane_location = prev_vehicle_list[objectID].lane_location
            
            vehicle = vehicle_list.get(id)
            if not vehicle:
                vehicle = Vehicle(vehicle_type, centroid_location, id, intersection_number, lane_location, action)
                vehicle_list[id] = vehicle
            else:

                vehicle.centroid_location = centroid_location
                vehicle.intersection_number = intersection_number
                vehicle.lane_location = lane_location


            emergency_vehicles = [v for v in vehicle_list.values() if v.vehicle_type == "emergency_vehicle"]
            normal_vehicles = [v for v in vehicle_list.values() if v.vehicle_type == "normal_vehicle"]
            if emergency_vehicles:
                emergency_vehicle = emergency_vehicles[0]  # Taking the first emergency vehicle found
                emergency_intersection_number = emergency_vehicle.intersection_number
                for vehicle in normal_vehicles:
                    if emergency_vehicle.lane_location == "left":
                        vehicle.action = instruction_delivery_Emergency_Vehicle_At_left(vehicle, emergency_intersection_number)
                    elif emergency_vehicle.lane_location == "stra":
                        vehicle.action = instruction_delivery_Emergency_Vehicle_At_middle(vehicle, emergency_intersection_number)
                    elif emergency_vehicle.lane_location == "right":
                        vehicle.action = instruction_delivery_Emergency_Vehicle_At_right(vehicle, emergency_intersection_number)
            text = f"ID {objectID}, {vehicle.intersection_number} {vehicle.lane_location}, {vehicle.action}"
            cv2.putText(frame, text, (centroid[0] - 35, centroid[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            
        

        for lane, lane_mask in intersection_annotation[frame_count].items():
            # polygon plotting code from https://www.geeksforgeeks.org/python-opencv-cv2-polylines-method/
            pts = np.array(lane_mask, np.int32)
 
            pts = pts.reshape((-1, 1, 2))
            
            isClosed = True
            # Red color in BGR
            color = (0, 0, 255)
            
            # Line thickness of 8 px
            thickness = 2
            
            # Using cv2.polylines() method
            # Draw a Green polygon with 
            # thickness of 1 px
            frame = cv2.polylines(frame, [pts], 
                                isClosed, color, 
                                thickness)

        prev_vehicle_list = vehicle_list

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
    yolo_counter('./yolo_files/coco.names', 'front_sample.mp4', "side_annotations.xml", 'output.mp4')

if __name__ == '__main__':
    main()