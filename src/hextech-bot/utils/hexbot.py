import cv2 as cv
import numpy as np
import pandas as pd

from utils.timer import timefunc
from utils.image_processing import prepare_image

from utils.square import Square
from utils.command import Command
from utils.command_type import CommandType


class HexBot():

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

    BUTTONS = {
        CommandType.NORM_JUMP: 'JUMP',
        CommandType.FAST_JUMP: 'JUMP',
        CommandType.NORM_DOWN: 'DOWN',
        CommandType.FAST_DOWN: 'DOWN',
        CommandType.NORM_BOMB: 'BOMB',
        CommandType.FAST_BOMB: 'BOMB',
    }

    def __init__(self, square_coords, method, threshold, target_x, velocity, time_window, time_offset):
        self.squares = [Square(xmin, xmax, ymin, ymax) for xmin, xmax, ymin, ymax in square_coords]

        self.threshold = threshold
        self.target_x = target_x
        self.velocity = velocity

        self.method = method

        self.time_window = time_window
        self.time_offset = time_offset

        self.queue_df = pd.DataFrame(columns=['Time', 'Command'])
        self.exec_df = pd.DataFrame(columns=['Frame', 'Command'])

    @timefunc(verbose=False)
    def process_frame(self, frame, time: float):
        commands = [self.process_square(frame, square, time) for square in self.squares]

        for cmd in commands:
            self.add_to_queue(cmd)

        self.queue_df.drop_duplicates(subset=['Time'], inplace=True)
        self.queue_df.sort_values(by=['Time'], inplace=True)

    def process_square(self, frame, square: Square, time: float):
        cut = square.cut_frame(frame)
        gray = prepare_image(cut)

        xmin = square.top_left.x
        cmd = self.find_cmd(gray)

        if cmd:
            cmd.calc_time(xmin, self.target_x, self.velocity, time)
            cut = cmd.mark(cut)

        height, width, _ = cut.shape
        cv.rectangle(cut, (0, 0), (width, height), (0, 0, 255), 2)

        return cmd

    def find_cmd(self, img) -> Command:

        results_val = []
        results = []

        for type in Command.TYPES:
            template = HexBot.TEMPLATES[type]
            mask = HexBot.MASK[type]

            val, top_left, bottom_right = self.find_template(img, template, mask)

            cmd = Command(type, val, top_left, bottom_right)

            results.append(cmd)
            results_val.append(val)

        max_ind = np.argmax(results_val)
        result = results[max_ind]

        if result.val > self.threshold:
            return result

        return None

    def find_template(self, img, template, mask):
        img = img.copy()

        # Apply template Matching
        res = cv.matchTemplate(img, template, self.method)  # mask=mask
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if self.method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            val = min_val
            top_left = min_loc
        else:
            val = max_val
            top_left = max_loc

        h, w = template.shape

        bottom_right = (top_left[0] + w, top_left[1] + h)
        return val, top_left, bottom_right

    def add_to_queue(self, cmd):
        if cmd:
            self.queue_df = self.queue_df.append({
                'Command': cmd.type,
                'Time': round(cmd.time, 1)
            }, ignore_index=True)

    def next_command(self, time: float) -> Command:
        tmin = time - self.time_offset - self.time_window
        tmax = time - self.time_offset + self.time_window
        query = f'Time >= {tmin} & Time <= {tmax}'
        queue_next = self.queue_df.query(query)

        if queue_next.shape[0] == 0:
            return None

        next_line = queue_next.iloc[0]
        name = next_line.name
        cmd = next_line['Command']

        self.queue_df.drop(name, inplace=True)

        return cmd

    def execute_command(self, n_frame, cmd):
        button = HexBot.BUTTONS[cmd]
        print(button)

        # TODO: press button

        self.exec_df = self.exec_df.append({
            'Frame': n_frame,
            'Command': cmd,
        }, ignore_index=True)

    def save_queue(self):
        self.queue_df.to_csv('data/queue.csv')
        print(self.queue_df)

    def save_exec(self):
        self.exec_df.to_csv('data/exec.csv')
        print(self.exec_df)
