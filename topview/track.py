import sys
sys.path.insert(0, './yolov5')

from yolov5.utils.google_utils import attempt_download
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh
from yolov5.utils.torch_utils import select_device, time_synchronized
from yolov5.utils.plots import plot_one_box
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from shapely.geometry import Point, Polygon
from core_logic import *
from lxml import etree
from vehicle import *

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

def detect_lane_location(outputs, confs, intersection_annotation, vehicle_list, prev_vehicle_list, frame_count, emergency_classes, im0, names):
    emergency_ids = []

    for j, (output, conf) in enumerate(zip(outputs, confs)): 
        bboxes = output[0:4]
        id = output[4]
        cls = output[5]

        c = int(cls)  # integer class

        centroid = (int((bboxes[0]+bboxes[2])/2), int((bboxes[1]+bboxes[3])/2))
        if id == 454:
            vehicle_type = "emergency_vehicle"
            emergency_ids.append(id)
            color = (0,0,255)
        else:
            vehicle_type = "normal_vehicle"
            color = (0,255,0)

        plot_one_box(bboxes, im0, label= str(int(id)), color=color, line_thickness=2)

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
        if intersection_number == " " and lane_location == " " and id in prev_vehicle_list:
            intersection_number = prev_vehicle_list[id].intersection_number
            lane_location = prev_vehicle_list[id].lane_location
        
        vehicle = Vehicle(vehicle_type, centroid, id, intersection_number, lane_location, action)
        vehicle_list[id] = vehicle
    
    return emergency_ids

def instruction_delivery(emergency_ids, vehicle_list):
    if emergency_ids:
        emergency_vehicle = vehicle_list[emergency_ids[0]]  # Taking the first emergency vehicle found
        emergency_intersection_number = emergency_vehicle.intersection_number
        for id in vehicle_list:
            if id in emergency_ids:
                continue
            if emergency_vehicle.lane_location == "left":
                instruction_delivery_Emergency_Vehicle_At_left(vehicle_list[id], emergency_intersection_number)
            elif emergency_vehicle.lane_location == "straight":
                instruction_delivery_Emergency_Vehicle_At_straight(vehicle_list[id], emergency_intersection_number)
            elif emergency_vehicle.lane_location == "right":
                instruction_delivery_Emergency_Vehicle_At_right(vehicle_list[id], emergency_intersection_number)
        
    else:
        for id in vehicle_list:
            vehicle_list[id].action = "continue"

def vehicle_display(frame, vehicle_list, show_id=True, show_location=True, show_action=True):
    for vehicle in vehicle_list.values():
        centroid = vehicle.centroid_location

        text = ""
        if show_id:
            text += f"ID {vehicle.ID}"
        if show_location:
            text += f", {vehicle.intersection_number} {vehicle.lane_location}"
        if show_action:
            text += f", {vehicle.action}"

        cv2.putText(frame, text, (centroid[0] - 35, centroid[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.circle(frame, (centroid[0], centroid[1]+10), 4, (0, 255, 0), -1)\

def plot_lanes(frame, intersection_annotation, frame_count, show_lanes):
    if show_lanes:
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

def compute_color_for_id(label):
    """
    Simple function that adds fixed color depending on the class
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def detect(opt):
    out, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate = \
        opt.output, opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
            opt.save_txt, opt.img_size, opt.evaluate
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')
    
    mask_path = opt.lane_annotation
    show_id = opt.show_id
    show_location = opt.show_location
    show_action = opt.show_action
    show_lanes = opt.show_lanes

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    save_path = str(Path(out))
    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(out)) + '/' + txt_file_name + '.txt'

    frame_count = 0
    bus_class_id = 5

    vehicle_list = {}
    prev_vehicle_list = {}

    intersection_annotation = parse_anno_file(mask_path)

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()


        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                    #if det[:, -1]
                    #s += f"{det[:, 0:4] == c}"

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, -1]

                # pass detections to deepsort
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss, im0)
                
                # draw boxes for visualization
                #print(outputs)
                t3 = time_synchronized()
                if len(outputs) > 0:
                    emergency_ids = detect_lane_location(outputs=outputs,
                                                         confs=confs,
                                                         intersection_annotation=intersection_annotation,
                                                         vehicle_list=vehicle_list,
                                                         prev_vehicle_list=prev_vehicle_list,
                                                         frame_count=frame_count,
                                                         emergency_classes=[bus_class_id],
                                                         im0=im0,
                                                         names=names)

                    instruction_delivery(emergency_ids=emergency_ids, vehicle_list=vehicle_list)
                    
                    vehicle_display(im0, vehicle_list, show_id, show_location, show_action)
                    
                    plot_lanes(im0, intersection_annotation, frame_count, show_lanes)
                    
            else:
                deepsort.increment_ages()
            
            t4 = time_synchronized()
            # Print time (inference + NMS)
            print('%sDone. yolo:(%.3fs) deepsort:(%.3fs) logic:(%.3fs) total:(%.3fs) fps:(%.3fs)' % (s, t2 - t1, t3 - t2, t4 - t3, t4-t1, 1/(t4-t1)))

            prev_vehicle_list = vehicle_list
            vehicle_list = {}
            frame_count += 1

            # Stream results
            if show_vid:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

    if save_txt or save_vid:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', type=str, default='yolov5/weights/yolov5m.pt', help='model.pt path')
    parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    parser.add_argument('--lane_annotation', type=str, help="lane annotation file")
    parser.add_argument('--show_id', action='store_true', help="show id on output video")
    parser.add_argument('--show_location', action='store_true', help="show lane location on output video")
    parser.add_argument('--show_action', action='store_true', help="show assigned instruction on output video")
    parser.add_argument('--show_lanes', action='store_true', help="show lanes on output video")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)

    with torch.no_grad():
        detect(args)
