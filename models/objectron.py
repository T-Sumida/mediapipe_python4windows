# -*- coding:utf-8
import cv2
import numpy as np
import mediapipe as mp
from loguru import logger

from .abst_detector import AbstDetector


class Objectron(AbstDetector):
    def __init__(self, max_num_objects: int, min_detection_confidence: float, min_tracking_confidence: float, model_name: str) -> None:
        """初期化処理

        Args:
            max_num_objects (int): 最大検出物体数
            min_detection_confidence (float): 物体検出モデルの最小信頼値
            min_tracking_confidence (float): ランドマーク追跡モデルからの最小信頼値
            model_name (str): モデル名
        """
        self.objectron = mp.solutions.objectron.Objectron(
            static_image_mode=False,
            max_num_objects=max_num_objects,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_name=model_name
        )

    def detect(self, image: np.ndarray) -> bool:
        """物体検出処理

        Args:
            image (np.ndarray): 入力イメージ

        Returns:
            bool: 対象物体が検出できたか
        """
        try:
            self.results = self.objectron.process(image)
        except Exception as e:
            logger.error(e)
        return True if self.results.detected_objects is not None else False

    def draw(self, image: np.ndarray) -> np.ndarray:
        """処理結果を描画する

        Args:
            image (np.ndarray): ベースイメージ

        Returns:
            np.ndarray: 描画済みイメージ
        """
        base_width, base_height = image.shape[1], image.shape[0]

        for detected_objects in self.results.detected_objects:
            landmark_buf = []

            # draw landmarks
            for landmark in detected_objects.landmarks_2d.landmark:
                x = min(int(landmark.x * base_width), base_width - 1)
                y = min(int(landmark.y * base_height), base_height - 1)
                landmark_buf.append((x, y))
                cv2.circle(image, (x, y), 3, (255, 0, 0), 5)

            # draw connections
            for con_pair in mp.solutions.objectron.BOX_CONNECTIONS:
                cv2.line(image, landmark_buf[con_pair[0].value],
                         landmark_buf[con_pair[1].value], (255, 0, 0), 2)

        return image
