import cv2 as cv
import numpy as np
import pandas as pd

import pyautogui

from utils.command import Command
from utils.command_type import CommandType
from utils.image_processing import prepare_image


pyautogui.PAUSE = 0


class HexBot:

    def __init__(self, region, method, threshold: float, interval: float):
        self.xmin, self.xmax, self.ymin, self.ymax = region
        self.method = method
        self.threshold = threshold
        self.interval = interval

        self.current_time = 0
        self.start_jump = 0

        self.queue_df = pd.DataFrame(columns=['Time', 'Command'])
        self.exec_df = pd.DataFrame(columns=['Time', 'Command'])

    def set_current_time(self, time: float):
        self.current_time = time - self.start_jump

    def process_frame(self, frame, target_x: float, velocity: float):
        cut = frame[self.ymin:self.ymax, self.xmin:self.xmax, :]

        cmd = self.find_cmd(cut, target_x, velocity)
        self.add_to_queue(cmd)

        self.clean_similar()

        height, width, _ = cut.shape
        cv.rectangle(cut, (0, 0), (width, height), (0, 0, 255), 2)

    def find_cmd(self, image, target_x, velocity) -> Command:
        results_val = []
        results = []

        gray = prepare_image(image)

        for type in Command.TYPES:
            template = prepare_image(Command.TEMPLATES[type])

            val, top_left, bottom_right = self.find_template(gray, template)
            cmd = Command(type, val, (self.xmin, self.ymin), top_left, bottom_right, target_x, velocity, self.current_time)

            results.append(cmd)
            results_val.append(val)

        max_ind = np.argmax(results_val)
        result = results[max_ind]

        if result.val > self.threshold and result.val <= 1:

            template = Command.TEMPLATES[result.cmd_type]
            mask = Command.MASK[result.cmd_type]

            val, _, _ = self.find_template(image, template, mask)

            if val > self.threshold:
                result.val = val
                return result

        return None

    def find_template(self, img, template, mask=None):
        img = img.copy()

        # Apply template Matching
        res = cv.matchTemplate(img, template, self.method, mask=mask)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if self.method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            val = 1 - min_val
            top_left = min_loc
        else:
            val = max_val
            top_left = max_loc

        h, w = template.shape[:2]

        bottom_right = (top_left[0] + w, top_left[1] + h)
        return val, top_left, bottom_right

    def add_to_queue(self, cmd: Command):
        if cmd:
            if self.start_jump == 0 and cmd.cmd_type == CommandType.NORM_JUMP:
                self.start_jump = cmd.time - (self.interval / 2)
                self.current_time -= self.start_jump
                cmd.time = 0

            self.queue_df = self.queue_df.append({
                'Time': cmd.time,
                'RoundTime': np.round(cmd.time / self.interval) * self.interval,
                'Command': cmd.cmd_type,
                'Value': cmd.val,
                'X': cmd.x,
                'Y': cmd.y,
            }, ignore_index=True)

    def clean_similar(self):
        self.queue_df.drop_duplicates(subset=['RoundTime', 'Command'], inplace=True)
        self.queue_df.sort_values(by=['Time'], inplace=True)

    def show_queue(self, frame, target_x: float, velocity: float):
        for _, row in self.queue_df.iterrows():
            cmd = row['Command']
            cmd_time = row['Time']
            y = int(row['Y'])
            val = row['Value']

            dt = cmd_time - self.current_time
            dx = velocity * dt
            x = int(target_x - dx)

            s = 20
            pt1 = (x - s, y - s)
            pt2 = (x + s, y + s)

            color = Command.COLOR[cmd]
            cv.rectangle(frame, pt1, pt2, color, 2)
            cv.putText(frame, f'{val:0.2f}', pt1, cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def next_command(self) -> Command:
        query = f'Time <= {self.current_time}'
        queue_next = self.queue_df.query(query)

        n = queue_next.shape[0]
        if n == 0:
            return None

        cmd_counts = queue_next['Command'].value_counts()
        cmd = cmd_counts.index[0]

        self.queue_df.drop(queue_next.index, inplace=True)
        return cmd

    def execute_command(self, cmd: CommandType):
        if cmd is None:
            return

        button = Command.BUTTONS[cmd]
        action = Command.ACTION_NAME[cmd]

        pyautogui.press(button)

        self.exec_df = self.exec_df.append({
            'Time': self.current_time,
            'Command': action,
        }, ignore_index=True)

    def save_exec(self, path: str = 'data/exec.csv'):
        self.exec_df.to_csv(path, index=None)
