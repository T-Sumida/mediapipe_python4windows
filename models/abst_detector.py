# -*- coding:utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np

class AbstDetector(metaclass=ABCMeta):
    @abstractmethod
    def detect(self, image) -> bool:
        pass
    
    @abstractmethod
    def show(self, image) -> np.array:
        pass