# -*- coding:utf-8
from enum import Enum
from typing import Text

import cv2
import numpy as np
import mediapipe as mp
from loguru import logger
from .abst_detector import AbstDetector


class Mode(Enum):
    USE_BG_IMAGE = 0
    USE_BLUR = 1


class SelfieSegmentation(AbstDetector):
    def __init__(self, model_selection: int = 0, bg_image_path: Text = None) -> None:
        """初期化処理

        Args:
            model_selection (int, optional): モデルタイプ {0, 1}. Defaults to 0.
            bg_image_path (Text, optional): 背景画像のパス. Defaults to None.
        """
        self.segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=model_selection)
        if cv2.imread(bg_image_path) is not None:
            self.mode = Mode.USE_BG_IMAGE
            self.bg_image = cv2.imread(bg_image_path)
        else:
            self.mode = Mode.USE_BLUR

    def detect(self, image: np.ndarray) -> bool:
        """セグメンテーション処理

        Args:
            image (np.ndarray): 入力イメージ

        Returns:
            bool: セグメンテーションマスクを取得できたかどうか
        """
        try:
            self.results = self.segmentation.process(image)
        except Exception as e:
            logger.error(e)
        return True if self.results.segmentation_mask is not None else False

    def draw(self, image: np.ndarray) -> np.ndarray:
        """処理結果を描画する

        Args:
            image (np.ndarray): ベースイメージ

        Returns:
            np.ndarray: 描画済みイメージ
        """
        if self.mode == Mode.USE_BG_IMAGE:
            self.bg_image = cv2.resize(
                self.bg_image, (image.shape[1], image.shape[0]))
        elif self.mode == Mode.USE_BLUR:
            self.bg_image = cv2.GaussianBlur(image, (55, 55), 0)

        condition = np.stack(
            (self.results.segmentation_mask,) * 3, axis=-1) > 0.1

        output_image = np.where(condition, image, self.bg_image)
        return output_image
