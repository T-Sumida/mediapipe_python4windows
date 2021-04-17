# -*- coding:utf-8
import logging
import cv2
import numpy as np
import mediapipe as mp
from loguru import logger

from .abst_detector import AbstDetector

class FaceDetector(AbstDetector):
    def __init__(self, min_detection_confidence: float) -> None:
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence)

    def detect(self, image) -> bool:
        try:
            self.results = self.face_detection.process(image)
        except Exception as e:
            logger.error(e)
        return True if self.results.detections is not None else False

    def show(self, image) -> np.array:
        for detection in self.results.detections:

            # face bounding box and id, score
            image = self.__draw_bounding_box_score(
                image,
                detection.location_data.relative_bounding_box,
                detection.label_id[0], detection.score[0]
            )

            # right eye keypoint
            image = self.__draw_key_points(image, detection.location_data.relative_keypoints[0])
            # left eye keypoint
            image = self.__draw_key_points(image, detection.location_data.relative_keypoints[1])
            # nose keypoint
            image = self.__draw_key_points(image, detection.location_data.relative_keypoints[2])
            # mouth keypoint
            image = self.__draw_key_points(image, detection.location_data.relative_keypoints[3])
            # right ear keypoint
            image = self.__draw_key_points(image, detection.location_data.relative_keypoints[4])
            # left ear keypoint
            image = self.__draw_key_points(image, detection.location_data.relative_keypoints[5])

        return image
    
    def __draw_bounding_box_score(self, image, bbox, detect_id, detect_score) -> np.array:
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
    
    def __draw_key_points(self, image, keypoint) -> np.array:
        base_width, base_height = image.shape[1], image.shape[0]
        x = int(keypoint.x * base_width)
        y = int(keypoint.y * base_height)

        cv2.circle(
            image, (x, y), 2, (255, 0, 0), 5
        )
        return image
