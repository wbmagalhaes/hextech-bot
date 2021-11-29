import cv2 as cv
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
        CommandType.NORM_JUMP: prepare_image(cv.imread('jump_template.png')),
        CommandType.FAST_JUMP: prepare_image(cv.imread('jump_fast_template.png')),
        CommandType.NORM_DOWN: prepare_image(cv.imread('down_template.png')),
        CommandType.FAST_DOWN: prepare_image(cv.imread('down_fast_template.png')),
        CommandType.NORM_BOMB: prepare_image(cv.imread('bomb_template.png')),
        CommandType.FAST_BOMB: prepare_image(cv.imread('bomb_fast_template.png')),
    }

    MASKS = {
        CommandType.NORM_JUMP: cv.imread('jump_mask.png', 0),
        CommandType.FAST_JUMP: cv.imread('jump_fast_mask.png', 0),
        CommandType.NORM_DOWN: cv.imread('down_mask.png', 0),
        CommandType.FAST_DOWN: cv.imread('down_fast_mask.png', 0),
        CommandType.NORM_BOMB: cv.imread('bomb_mask.png', 0),
        CommandType.FAST_BOMB: cv.imread('bomb_fast_mask.png', 0),
    }

    COLOR = {
        CommandType.NORM_JUMP: (0, 255, 0),
        CommandType.FAST_JUMP: (0, 255, 0),
        CommandType.NORM_DOWN: (255, 255, 255),
        CommandType.FAST_DOWN: (255, 255, 255),
        CommandType.NORM_BOMB: (255, 0, 0),
        CommandType.FAST_BOMB: (255, 0, 0),
    }

    def __init__(self, type: CommandType, val: float, top_left: tuple, bottom_right: tuple) -> None:
        self.type = type
        self.val = val
        self.top_left = top_left
        self.bottom_right = bottom_right

    def getColor(self):
        return Command.COLOR[self.type]
