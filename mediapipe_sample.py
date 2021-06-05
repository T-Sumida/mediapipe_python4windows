# -*- coding:utf-8 -*-
import copy
import argparse

import cv2
import numpy as np
from loguru import logger

import models
from utils import FpsCalculator


def get_args() -> argparse.Namespace:
    """引数取得

    Returns:
        argparse.Namespace: 取得結果
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", help='device id', type=int, default=0)
    parser.add_argument("--width", help='capture width', type=int, default=960)
    parser.add_argument(
        "--height", help='capture height', type=int, default=540
    )

    subparsers = parser.add_subparsers(dest="model")

    # face_detection command parser
    parser_fd = subparsers.add_parser(
        'FaceDetector', help='', description='FaceDetector'
    )
    parser_fd.add_argument(
        '--min_detection_confidence', type=float,
        default=0.7, help='顔検出モデルの最小信頼値 [0.0, 1.0]'
    )

    # face_mesh command parser
    parser_fm = subparsers.add_parser(
        'FaceMesh', help='', description='FaceMesh'
    )
    parser_fm.add_argument(
        '--max_num_faces', type=int, default=2, help='最大検出顔数'
    )
    parser_fm.add_argument(
        '--min_detection_confidence', type=float, default=0.7,
        help='顔検出モデルの最小信頼値 [0.0, 1.0]'
    )
    parser_fm.add_argument(
        '--min_tracking_confidence', type=float, default=0.5,
        help='ランドマーク追跡モデルの最小信頼値 [0.0, 1.0]'
    )

    # hand_tracker command parser
    parser_ht = subparsers.add_parser(
        'HandTracker', help='', description='HandTracker'
    )
    parser_ht.add_argument(
        '--max_num_hands', type=int, default=2, help='最大検出手数'
    )
    parser_ht.add_argument(
        '--min_detection_confidence', type=float, default=0.7,
        help='手検出モデルの最小信頼値 [0.0, 1.0]'
    )
    parser_ht.add_argument(
        '--min_tracking_confidence', type=float, default=0.5,
        help='ランドマーク追跡モデルの最小信頼値 [0.0, 1.0]'
    )

    # pose_estimator command parser
    parser_pe = subparsers.add_parser(
        'PoseEstimator', help='', description='PoseEstimator'
    )
    parser_pe.add_argument(
        '--min_detection_confidence', type=float, default=0.7,
        help='姿勢推定モデルの最小信頼値 [0.0, 1.0]'
    )
    parser_pe.add_argument(
        '--min_tracking_confidence', type=float, default=0.5,
        help='ランドマーク追跡モデルの最小信頼値 [0.0, 1.0]'
    )

    # objectron command parser
    parser_ob = subparsers.add_parser(
        'Objectron', help='', description='Objectron')
    parser_ob.add_argument('--max_num_objects', type=int,
                           default=5, help='最大検出物体数')
    parser_ob.add_argument('--min_detection_confidence',
                           type=float, default=0.7,
                           help='物体検出モデルの最小信頼値 [0.0, 1.0]')
    parser_ob.add_argument('--min_tracking_confidence',
                           type=float, default=0.5,
                           help='ランドマーク追跡モデルの最小信頼値 [0.0, 1.0]')
    parser_ob.add_argument('--model_name', type=str,
                           default='Chair',
                           help='モデル名 {Shoe, Chair, Cup, Camera}')

    # holistic command parser
    parser_pe = subparsers.add_parser(
        'Holistic', help='', description='Holistic')
    parser_pe.add_argument('--min_detection_confidence',
                           type=float, default=0.7,
                           help='人物検出モデルの最小信頼値 [0.0, 1.0]')
    parser_pe.add_argument('--min_tracking_confidence',
                           type=float, default=0.5,
                           help='ランドマーク追跡モデルの最小信頼値 [0.0, 1.0]')

    # SelfieSegmentation command parser
    parser_pe = subparsers.add_parser(
        'SelfieSegmentation', help='', description='SelfieSegmentation')
    parser_pe.add_argument('--model_selection',
                           type=int, default=0,
                           help='モデルのタイプ 0(general model) or 1(fast model)')
    parser_pe.add_argument('--bg_image_path',
                           type=str, default=None,
                           help='背景画像のパス')

    args = parser.parse_args()
    return args


def draw_fps(image: np.ndarray, fps: int) -> np.ndarray:
    """fpsを描画する

    Args:
        image (np.ndarray): ベースイメージ
        fps (int): FPS

    Returns:
        np.ndarray: 描画済みイメージ
    """
    width = image.shape[1]

    cv2.rectangle(image, (width-105, 0), (width, 20), (0, 0, 0), -1)
    cv2.putText(image, "FPS: " + str(fps),
                (width-100, 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1)
    return image


def main() -> None:
    """メインループ"""
    args = get_args().__dict__

    # setting camera device
    cap = cv2.VideoCapture(args['device'])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args['height'])
    del args['device'], args['width'], args['height']

    # setting detector
    try:
        model_name = args['model']
        del args['model']
        detector = getattr(models, model_name)(**args)
    except Exception as e:
        logger.error(e)
        exit(1)

    # setting fps calculator
    calculator = FpsCalculator()

    # main loop
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break

        image = cv2.flip(image, 1)
        tmp_image = copy.deepcopy(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if detector.detect(image):
            tmp_image = detector.draw(tmp_image)

        fps = calculator.calc()
        tmp_image = draw_fps(tmp_image, fps)
        cv2.imshow(model_name, tmp_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return


if __name__ == "__main__":
    main()
