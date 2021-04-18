# -*- coding:utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np


class AbstDetector(metaclass=ABCMeta):
    @abstractmethod
    def detect(self, image: np.ndarray) -> bool:
        """モデルによる推論処理

        Args:
            image (np.ndarray): 入力イメージ

        Returns:
            bool: 対象物体（手や顔）が検出できたかどうか
        """
        pass

    @abstractmethod
    def draw(self, image: np.ndarray) -> np.ndarray:
        """推論結果を描画する

        Args:
            image (np.ndarray): ベースイメージ

        Returns:
            np.ndarray: 描画済みイメージ
        """
        pass
