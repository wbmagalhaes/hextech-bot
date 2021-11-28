import cv2 as cv
from enum import Enum


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
        CommandType.NORM_JUMP: cv.imread('norm_jump_template.png', 0),
        CommandType.FAST_JUMP: cv.imread('fast_jump_template.png', 0),
        CommandType.NORM_DOWN: cv.imread('norm_down_template.png', 0),
        CommandType.FAST_DOWN: cv.imread('norm_down_template.png', 0),
        CommandType.NORM_BOMB: cv.imread('norm_bomb_template.png', 0),
        CommandType.FAST_BOMB: cv.imread('fast_bomb_template.png', 0),
    }

    MASKS = {
        CommandType.NORM_JUMP: cv.imread('norm_jump_template.png', 0),
        CommandType.FAST_JUMP: cv.imread('fast_jump_template.png', 0),
        CommandType.NORM_DOWN: cv.imread('norm_down_template.png', 0),
        CommandType.FAST_DOWN: cv.imread('norm_down_template.png', 0),
        CommandType.NORM_BOMB: cv.imread('norm_bomb_template.png', 0),
        CommandType.FAST_BOMB: cv.imread('fast_bomb_template.png', 0),
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
