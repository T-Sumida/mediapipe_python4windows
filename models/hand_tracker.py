# -*- coding:utf-8
import cv2
import numpy as np
import mediapipe as mp
from loguru import logger

from .abst_detector import AbstDetector


class HandTracker(AbstDetector):
    def __init__(self, max_num_hands: int, min_detection_confidence: float, min_tracking_confidence: float) -> None:
        """初期化処理

        Args:
            max_num_hands (int): 最大検出手数
            min_detection_confidence (float): 手検出モデルの最小信頼値
            min_tracking_confidence (float): ランドマーク追跡モデルからの最小信頼値
        """
        self.tracker = mp.solutions.hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, image: np.ndarray) -> bool:
        """手検出処理

        Args:
            image (np.ndarray): 入力イメージ

        Returns:
            bool: 手が検出できたか
        """
        try:
            self.results = self.tracker.process(image)
        except Exception as e:
            logger.error(e)
        return True if self.results.multi_hand_landmarks is not None else False

    def draw(self, image: np.ndarray) -> np.ndarray:
        """処理結果を描画する

        Args:
            image (np.ndarray): ベースイメージ

        Returns:
            np.ndarray: 描画済みイメージ
        """
        base_width, base_height = image.shape[1], image.shape[0]
        for hand_landmarks, handedness in zip(self.results.multi_hand_landmarks, self.results.multi_handedness):

            landmark_buf = []

            # keypoint
            for landmark in hand_landmarks.landmark:
                x = min(int(landmark.x * base_width), base_width - 1)
                y = min(int(landmark.y * base_height), base_height - 1)
                landmark_buf.append((x, y))
                cv2.circle(image, (x, y), 3, (255, 0, 0), 5)

            # connection line
            for con_pair in mp.solutions.hands.HAND_CONNECTIONS:
                cv2.line(image, landmark_buf[con_pair[0].value],
                         landmark_buf[con_pair[1].value], (255, 0, 0), 2)

        return image
