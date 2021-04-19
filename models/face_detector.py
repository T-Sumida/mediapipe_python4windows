# -*- coding:utf-8
import cv2
import numpy as np
import mediapipe as mp
from loguru import logger
from mediapipe.framework.formats.location_data_pb2 import _LOCATIONDATA_RELATIVEBOUNDINGBOX, _LOCATIONDATA_RELATIVEKEYPOINT

from .abst_detector import AbstDetector


class FaceDetector(AbstDetector):
    def __init__(self, min_detection_confidence: float) -> None:
        """初期化処理

        Args:
            min_detection_confidence (float): 顔検出モデルの最小信頼値
        """
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence)

    def detect(self, image: np.ndarray) -> bool:
        """顔検出処理

        Args:
            image (np.ndarray): 入力イメージ

        Returns:
            bool: 顔が検出できたか
        """
        try:
            self.results = self.face_detection.process(image)
        except Exception as e:
            logger.error(e)
        return True if self.results.detections is not None else False

    def draw(self, image: np.ndarray) -> np.ndarray:
        """処理結果を描画する

        Args:
            image (np.ndarray): ベースイメージ

        Returns:
            np.ndarray: 描画済みイメージ
        """
        for detection in self.results.detections:

            # face bounding box and id, score
            image = self.__draw_bounding_box_score(
                image,
                detection.location_data.relative_bounding_box,
                detection.label_id[0], detection.score[0]
            )

            # right eye keypoint
            image = self.__draw_key_points(
                image, detection.location_data.relative_keypoints[0])
            # left eye keypoint
            image = self.__draw_key_points(
                image, detection.location_data.relative_keypoints[1])
            # nose keypoint
            image = self.__draw_key_points(
                image, detection.location_data.relative_keypoints[2])
            # mouth keypoint
            image = self.__draw_key_points(
                image, detection.location_data.relative_keypoints[3])
            # right ear keypoint
            image = self.__draw_key_points(
                image, detection.location_data.relative_keypoints[4])
            # left ear keypoint
            image = self.__draw_key_points(
                image, detection.location_data.relative_keypoints[5])

        return image

    def __draw_bounding_box_score(self, image: np.ndarray, bbox: _LOCATIONDATA_RELATIVEBOUNDINGBOX, detect_id: int, detect_score: float) -> np.ndarray:
        """バウンディングボックスとラベル・スコアを描画する

        Args:
            image (np.ndarray): ベースイメージ
            bbox (_LOCATIONDATA_RELATIVEBOUNDINGBOX): バウンディングボックス情報
            detect_id (int): 検出ID
            detect_score (float): 信頼値

        Returns:
            np.ndarray: 描画済みイメージ
        """
        base_width, base_height = image.shape[1], image.shape[0]
        xmin = int(bbox.xmin * base_width)
        ymin = int(bbox.ymin * base_height)
        width = int(bbox.width * base_width)
        height = int(bbox.height * base_height)
        cv2.rectangle(
            image, (xmin, ymin),
            ((xmin + width), (ymin + height)),
            (255, 0, 0), 2
        )
        cv2.putText(
            image,
            "ID:" + str(detect_id) + ", Score:" + str(round(detect_score, 3)),
            (xmin, (ymin + height + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
            (255, 0, 0), 2, cv2.LINE_AA)
        return image

    def __draw_key_points(self, image: np.ndarray, keypoint: _LOCATIONDATA_RELATIVEKEYPOINT) -> np.ndarray:
        """キーポイントを描画する

        Args:
            image (np.ndarray): ベースイメージ
            keypoint (_LOCATIONDATA_RELATIVEKEYPOINT): 顔のキーポイント情報

        Returns:
            np.ndarray: 描画済みイメージ
        """
        base_width, base_height = image.shape[1], image.shape[0]
        x = int(keypoint.x * base_width)
        y = int(keypoint.y * base_height)

        cv2.circle(
            image, (x, y), 2, (255, 0, 0), 5
        )
        return image
