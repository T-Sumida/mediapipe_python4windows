# -*- coding:utf-8
import cv2
import numpy as np
import mediapipe as mp
from loguru import logger

from .abst_detector import AbstDetector


class PoseEstimator(AbstDetector):
    def __init__(self, min_detection_confidence: float, min_tracking_confidence: float) -> None:
        self.estimator = mp.solutions.pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, image) -> bool:
        try:
            self.results = self.estimator.process(image)
        except Exception as e:
            logger.error(e)
        return True if self.results.pose_landmarks is not None else False

    def draw(self, image) -> np.array:
        landmark_buf = []
        base_width, base_height = image.shape[1], image.shape[0]

        # draw landmark points
        for landmark in self.results.pose_landmarks.landmark:
            x = min(int(landmark.x * base_width), base_width - 1)
            y = min(int(landmark.y * base_height), base_height - 1)
            landmark_buf.append((x, y))
            cv2.circle(image, (x, y), 2, (255, 0, 0), 5)

        # draw connections
        for con_pair in mp.solutions.pose.POSE_CONNECTIONS:
            cv2.line(image, landmark_buf[con_pair[0].value],
                     landmark_buf[con_pair[1].value], (255, 0, 0), 2)

        return image
