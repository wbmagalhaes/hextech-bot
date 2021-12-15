from enum import Enum


class CommandType(Enum):
    NORM_JUMP = 1
    FAST_JUMP = 2
    NORM_DOWN = 3
    FAST_DOWN = 4
    NORM_BOMB = 5
    FAST_BOMB = 6
