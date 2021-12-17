import cv2 as cv

from utils.command_type import CommandType
from utils.image_processing import prepare_image


class Command:

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

    ACTION_NAME = {
        CommandType.NORM_JUMP: 'JUMP',
        CommandType.FAST_JUMP: 'JUMP',
        CommandType.NORM_DOWN: 'DOWN',
        CommandType.FAST_DOWN: 'DOWN',
        CommandType.NORM_BOMB: 'BOMB',
        CommandType.FAST_BOMB: 'BOMB',
    }

    BUTTONS = {
        CommandType.NORM_JUMP: 'W',
        CommandType.FAST_JUMP: 'W',
        CommandType.NORM_DOWN: 'S',
        CommandType.FAST_DOWN: 'S',
        CommandType.NORM_BOMB: 'F',
        CommandType.FAST_BOMB: 'F',
    }

    def __init__(self, cmd_type: CommandType, val: float, ref_xy: tuple, top_left: tuple, bottom_right: tuple, target_x: float, velocity: float, time: float) -> None:
        self.cmd_type = cmd_type
        self.val = val

        self.x = (top_left[0] + bottom_right[0]) / 2 + ref_xy[0]
        self.y = (top_left[1] + bottom_right[1]) / 2 + ref_xy[1]

        dx = target_x - self.x
        dt = dx / velocity

        self.time = time + dt
