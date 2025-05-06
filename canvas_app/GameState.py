"""
@author: Trinh Khai Truong 
"""
import numpy as np
import random


class GameState:
    def __init__(self, classes, max_rounds=6, max_time=20):
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.image = np.zeros((480, 840, 3), dtype=np.uint8)
        self.classes_to_draw = random.sample(classes, max_rounds)
        self.current_round = 0
        self.current_class = self.classes_to_draw[0]
        self.predicted_class = ""
        self.predicted_confidence = 0.0
        self.user_drawings = []
        self.game_started = False
        self.drawing_started = False
        self.start_time = None
        self.max_time = max_time
        self.max_rounds = max_rounds
        