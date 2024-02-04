
import cv2
import numpy as np
import argparse
from vehicle import *
from core_logic import *
from lane_location import *
from object_detection_and_tracking import *


def main():
        parser = argparse.ArgumentParser(description='Process a video address')
        parser.add_argument('video_address', type=str, help='The address (URL or file path) of the video')
        args = parser.parse_args()
        print(f"Received video address: {args.video_address}")

        # ToDo: object detection and tracking

        # ToDo: lane location identification

        # ToDo: core logic (instruction delivery)

        

if __name__ == '__main__':
    main()

