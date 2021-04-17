# -*- coding:utf-8 -*-
import copy
import argparse

import cv2
from loguru import logger

import models
from utils import DispFps

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", help='device id', type=int, default=0)
    parser.add_argument("--width", help='capture width', type=int, default=960)
    parser.add_argument("--height", help='capture height', type=int, default=540)

    subparsers = parser.add_subparsers(dest="model")

    # face_detection command parser
    parser_fd = subparsers.add_parser('FaceDetector', help='', description='face detection')
    parser_fd.add_argument('--min_detection_confidence', type=float, default=0.7, help='fdsafdsa')

    # face_mesh command parser
    parser_fm = subparsers.add_parser('FaceMesh', help='', description='face mesh')
    parser_fm.add_argument('-mfdaaaae', type=float, default=0.7, help='faaaasa')

    # hand_tracker command parser
    parser_ht = subparsers.add_parser('HandTracker', help='', description='face mesh')
    parser_ht.add_argument('--max_num_hands', type=int, default=2, help='hand num')
    parser_ht.add_argument('--min_detection_confidence', type=float, default=0.7, help='-min_detection_confidence')
    parser_ht.add_argument('--min_tracking_confidence', type=float, default=0.5, help='min_tracking_confidence')
    
    args = parser.parse_args()
    return args


def main():
    args = get_args().__dict__

    # setting camera device
    cap = cv2.VideoCapture(args['device'])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args['height'])
    del args['device'], args['width'], args['height']

    # setting detector
    model_name = args['model']
    del args['model']
    detector = getattr(models, model_name)(**args)
    cap = cv2.VideoCapture(0)
    disp_fps = DispFps()

    # main loop
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break

        image = cv2.flip(image, 1)
        tmp_image = copy.deepcopy(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if detector.detect(image):
            tmp_image = detector.show(tmp_image)

        disp_fps.disp(tmp_image)
        cv2.imshow(model_name, tmp_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return


if __name__ == "__main__":
    main()