# -*- coding:utf-8
import cv2
import numpy as np
import mediapipe as mp
from loguru import logger

from .abst_detector import AbstDetector


class Holistic(AbstDetector):
    def __init__(self, min_detection_confidence: float, min_tracking_confidence: float) -> None:
        """初期化処理

        Args:
            min_detection_confidence (float): 人物検出モデルの最小信頼値
            min_tracking_confidence (float): ランドマーク追跡モデルからの最小信頼値
        """
        self.holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def detect(self, image: np.ndarray) -> bool:
        """人物検出処理

        Args:
            image (np.ndarray): 入力イメージ

        Returns:
            bool: 人物が検出できたか
        """
        try:
            self.results = self.holistic.process(image)
        except Exception as e:
            logger.error(e)
        return True

    def draw(self, image: np.ndarray) -> np.ndarray:
        """処理結果を描画する

        Args:
            image (np.ndarray): ベースイメージ

        Returns:
            np.ndarray: 描画済みイメージ
        """
        # face mesh
        if self.results.face_landmarks:
            self.__draw_keypoints_connections(image, self.results.face_landmarks.landmark,
                                              circle_radius=1, circle_size=1, connections=mp.solutions.face_mesh.FACE_CONNECTIONS)

        # pose
        if self.results.pose_landmarks:
            self.__draw_keypoints_connections(
                image, self.results.pose_landmarks.landmark, circle_radius=5, connections=mp.solutions.pose.POSE_CONNECTIONS)

        # left hand pose
        if self.results.left_hand_landmarks:
            self.__draw_keypoints_connections(image, self.results.left_hand_landmarks.landmark,
                                              circle_radius=5, connections=mp.solutions.holistic.HAND_CONNECTIONS)

        # right hand pose
        if self.results.right_hand_landmarks:
            self.__draw_keypoints_connections(image, self.results.right_hand_landmarks.landmark,
                                              circle_radius=5, connections=mp.solutions.holistic.HAND_CONNECTIONS)

        return image

    def __draw_keypoints_connections(self, image: np.ndarray, landmarks, circle_radius: int = 1, circle_size: int = 2, connections=None) -> np.ndarray:
        """キーポイントと連結部分を描画する

        Args:
            image (np.ndarray): ベースイメージ
            landmarks ([type]):ランドマーク情報
            circle_radius (int, optional): 描画円の半径サイズ. Defaults to 1.
            circle_size (int, optional): 描画円の太さ. Defaults to 2.
            connections ([type], optional): 連結情報. Defaults to None.

        Returns:
            np.ndarray: 描画済みイメージ
        """
        landmark_buf = []
        base_width, base_height = image.shape[1], image.shape[0]

        for landmark in landmarks:
            x = min(int(landmark.x * base_width), base_width - 1)
            y = min(int(landmark.y * base_height), base_height - 1)
            landmark_buf.append((x, y))
            cv2.circle(image, (x, y), circle_radius, (255, 0, 0), circle_size)

        if connections is not None:
            for con_pair in connections:
                cv2.line(image, landmark_buf[con_pair[0]],
                         landmark_buf[con_pair[1]], (255, 0, 0), 2)

        return image
