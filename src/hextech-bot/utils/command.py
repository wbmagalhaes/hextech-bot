import cv2 as cv

from utils.command_type import CommandType


class Command():

    TYPES = [type for type in CommandType]

    COLOR = {
        CommandType.NORM_JUMP: (9, 255, 185),
        CommandType.FAST_JUMP: (9, 255, 185),
        CommandType.NORM_DOWN: (245, 244, 246),
        CommandType.FAST_DOWN: (245, 244, 246),
        CommandType.NORM_BOMB: (247, 243, 24),
        CommandType.FAST_BOMB: (247, 243, 24),
    }

    def __init__(self, type: CommandType, val: float, top_left: tuple, bottom_right: tuple) -> None:
        self.type = type
        self.val = val

        self.top_left = top_left
        self.bottom_right = bottom_right

        self.color = Command.COLOR[type]
        self.time = 0

    def calc_time(self, min_x: float, target_x: float, velocity: float, time: float):
        x = min_x + (self.top_left[0] + self.bottom_right[0]) / 2

        dx = target_x - x
        dt = dx / velocity

        self.time = time + dt

    def mark(self, img):
        cv.rectangle(img, self.top_left, self.bottom_right, self.color, 1)
        cv.putText(img, f'{self.val:0.2f}', self.top_left, cv.FONT_HERSHEY_SIMPLEX, 0.4, self.color, 1)
        return img
