# -*- coding:utf-8
import cv2
import numpy as np
import mediapipe as mp
from loguru import logger

from .abst_detector import AbstDetector


class FaceMesh(AbstDetector):
    def __init__(
        self, max_num_faces: int,
        min_detection_confidence: float,
        min_tracking_confidence: float
    ) -> None:
        """初期化処理

        Args:
            max_num_faces (int): 最大検出顔数
            min_detection_confidence (float): 顔検出モデルの最小信頼値
            min_tracking_confidence (float): ランドマーク追跡モデルからの最小信頼値
        """
        self.mesher = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, image: np.ndarray) -> bool:
        """顔検出処理

        Args:
            image (np.ndarray): 入力イメージ

        Returns:
            bool: 顔が検出できたか
        """
        try:
            self.results = self.mesher.process(image)
        except Exception as e:
            logger.error(e)
        return True if self.results.multi_face_landmarks is not None else False

    def draw(self, image: np.ndarray) -> np.ndarray:
        """処理結果を描画する

        Args:
            image (np.ndarray): ベースイメージ

        Returns:
            np.ndarray: 描画済みイメージ
        """
        landmark_buf = []
        base_width, base_height = image.shape[1], image.shape[0]
        for face_landmarks in self.results.multi_face_landmarks:

            # draw landmark points
            for landmark in face_landmarks.landmark:
                if landmark.visibility < 0 or landmark.presence < 0:
                    continue
                x = min(int(landmark.x * base_width), base_width - 1)
                y = min(int(landmark.y * base_height), base_height - 1)
                landmark_buf.append((x, y))
                cv2.circle(image, (x, y), 1, (255, 0, 0), 1)

            # draw connections
            for con_pair in mp.solutions.face_mesh.FACE_CONNECTIONS:
                cv2.line(
                    image, landmark_buf[con_pair[0]],
                    landmark_buf[con_pair[1]], (255, 0, 0), 2
                )

        return image
