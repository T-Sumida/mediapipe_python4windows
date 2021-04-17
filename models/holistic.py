# -*- coding:utf-8
import cv2
import numpy as np
import mediapipe as mp
from loguru import logger

from .abst_detector import AbstDetector


class Holistic(AbstDetector):
    def __init__(self, min_detection_confidence: float, min_tracking_confidence: float) -> None:
        self.holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def detect(self, image) -> bool:
        try:
            self.results = self.holistic.process(image)
        except Exception as e:
            logger.error(e)
        return True

    def draw(self, image) -> np.array:

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

    def __draw_keypoints_connections(self, image, landmarks, circle_radius=1, circle_size=2, connections=None) -> np.array:
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
