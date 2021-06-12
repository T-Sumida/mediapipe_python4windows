# -*- coding:utf-8 -*-
from collections import deque
import cv2


class FpsCalculator(object):
    def __init__(self, buffer_len=30):
        """Initialize

        Args:
            buffer_len (int, optional): time buffer size. Defaults to 30.
        """
        self._start_tick = cv2.getTickCount()
        self._freq = 1000.0 / cv2.getTickFrequency()
        self._difftimes = deque(maxlen=buffer_len)

    def calc(self) -> float:
        """calc fps

        Returns:
            float: FPS
        """
        current_tick = cv2.getTickCount()
        different_time = (current_tick - self._start_tick) * self._freq
        self._start_tick = current_tick

        self._difftimes.append(different_time)

        fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
        fps_rounded = round(fps, 2)

        return fps_rounded
