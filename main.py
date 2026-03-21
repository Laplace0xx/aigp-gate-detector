import cv2
import torch 
import time
import numpy as np
from abc import ABC, abstractmethod

class Detect(ABC):
    def __init__(self, roi_x: int = 0, roi_y: int = 0):
        self.roi_x = roi_x
        self.roi_y = roi_y
        self.latency_ms = 0.0

    @abstractmethod
    def detect(self, frame):
        ...

class DetectGate(Detect):
    def __init__(self, roi_x: int = 0, roi_y: int = 0):
        super().__init__(roi_x, roi_y)

