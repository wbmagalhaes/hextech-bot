import cv2 as cv
import numpy as np

from enum import Enum

from utils.image_processing import prepare_image


class CommandType(Enum):
    NORM_JUMP = 1
    FAST_JUMP = 2
    NORM_DOWN = 3
    FAST_DOWN = 4
    NORM_BOMB = 5
    FAST_BOMB = 6


class Command():

    TYPES = [type for type in CommandType]

    TEMPLATES = {
        CommandType.NORM_JUMP: prepare_image(cv.imread('data/jump_template.png')),
        CommandType.FAST_JUMP: prepare_image(cv.imread('data/jump_fast_template.png')),
        CommandType.NORM_DOWN: prepare_image(cv.imread('data/down_template.png')),
        CommandType.FAST_DOWN: prepare_image(cv.imread('data/down_fast_template.png')),
        CommandType.NORM_BOMB: prepare_image(cv.imread('data/bomb_template.png')),
        CommandType.FAST_BOMB: prepare_image(cv.imread('data/bomb_fast_template.png')),
    }

    MASK = {
        CommandType.NORM_JUMP: cv.imread('data/jump_mask.png', 0),
        CommandType.FAST_JUMP: cv.imread('data/jump_fast_mask.png', 0),
        CommandType.NORM_DOWN: cv.imread('data/down_mask.png', 0),
        CommandType.FAST_DOWN: cv.imread('data/down_fast_mask.png', 0),
        CommandType.NORM_BOMB: cv.imread('data/bomb_mask.png', 0),
        CommandType.FAST_BOMB: cv.imread('data/bomb_fast_mask.png', 0),
    }

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

    def calc_time(self, min_x, target_x, velocity, time):
        x = min_x + (self.top_left[0] + self.bottom_right[0]) / 2

        dx = target_x - x
        dt = dx / velocity

        self.time = time + dt

    def mark(self, img):
        cv.rectangle(img, self.top_left, self.bottom_right, self.color, 1)
        cv.putText(img, f'{self.val:0.2f}', self.top_left, cv.FONT_HERSHEY_SIMPLEX, 0.4, self.color, 1)
        return img

    @staticmethod
    def find_template(img, template, mask, method):
        img = img.copy()

        # Apply template Matching
        res = cv.matchTemplate(img, template, method)  # mask=mask
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            val = min_val
            top_left = min_loc
        else:
            val = max_val
            top_left = max_loc

        h, w = template.shape

        bottom_right = (top_left[0] + w, top_left[1] + h)
        return val, top_left, bottom_right

    @staticmethod
    def find(img, min_x, target_x, velocity, time, threshold=0.8, method=cv.TM_CCOEFF_NORMED):

        results_val = []
        results = []

        for type in Command.TYPES:
            template = Command.TEMPLATES[type]
            mask = Command.MASK[type]

            val, top_left, bottom_right = Command.find_template(img, template, mask, method=method)

            cmd = Command(type, val, top_left, bottom_right)
            cmd.calc_time(min_x, target_x, velocity, time)

            results.append(cmd)
            results_val.append(val)

        max_ind = np.argmax(results_val)
        result = results[max_ind]

        if result.val > threshold:
            return result

        return None
