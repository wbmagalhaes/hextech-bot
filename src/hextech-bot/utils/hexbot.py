import pyautogui

import cv2 as cv
import numpy as np
import pandas as pd

from utils.command import Command, CommandType


class HexBot:

    xmin, xmax, ymin, ymax = (0, 0, 0, 0)
    method = cv.TM_CCOEFF_NORMED
    threshold = 0.80
    interval = 0

    current_time = 0
    start_jump = 0

    queue_df = pd.DataFrame(columns=['Time', 'Command'])
    exec_df = pd.DataFrame(columns=['Time', 'Command'])

    def __init__(self, region, method, threshold: float, interval: float):
        self.xmin, self.xmax, self.ymin, self.ymax = region
        self.method = method
        self.threshold = threshold
        self.interval = interval

        pyautogui.PAUSE = 0

    def set_current_time(self, time: float):
        self.current_time = time - self.start_jump

    def process_frame(self, frame, target_x: float, velocity: float, verbose=False) -> Command:
        cut = frame[self.ymin:self.ymax, self.xmin:self.xmax, :]

        cmd = self.find_cmd(cut, target_x, velocity, verbose)
        self.add_to_queue(cmd)

        self.clean_similar()

        height, width, _ = cut.shape
        cv.rectangle(cut, (0, 0), (width, height), (0, 0, 255), 2)

        return cmd

    @staticmethod
    def prepare_image(img):
        # img_gray = img.copy()
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # img_gray = cv.Canny(img, 0, 255)
        return img_gray

    def find_cmd_type(self, type, img, target_x, velocity, verbose=False) -> Command:
        template = HexBot.prepare_image(Command.TEMPLATES[type])
        mask = Command.MASK[type]

        val, top_left, bottom_right = self.find_template(img, template, mask)
        cmd = Command(type, val, (self.xmin, self.ymin), top_left, bottom_right, target_x, velocity, self.current_time)

        if verbose:
            print(cmd.name, cmd.val)

        return cmd

    def confirm_comand(self, img, type: CommandType, verbose=False) -> float:
        template = Command.TEMPLATES[type]
        mask = Command.MASK[type]
        val, _, _ = self.find_template(img, template, mask)

        if verbose:
            print("Confirmation:", type, val)

        return val

    def find_cmd(self, image, target_x, velocity, verbose=False) -> Command:
        gray = HexBot.prepare_image(image)

        results = [self.find_cmd_type(type, gray, target_x, velocity, verbose) for type in Command.TYPES]
        max_ind = np.argmax([result.val for result in results])
        result = results[max_ind]

        if result.val > self.threshold and result.val <= 1:
            val = self.confirm_comand(image, result.cmd_type, verbose)
            if val > self.threshold:
                result.val = val
            return result

        return None

    def find_template(self, img, template, mask=None):
        img = img.copy()

        res = cv.matchTemplate(img, template, self.method, mask=mask)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

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
        if cmd is None:
            return

        # se é o primeiro comando, grava o offset de tempo
        if self.start_jump == 0 and cmd.cmd_type == CommandType.NORM_JUMP:
            self.start_jump = cmd.time - (self.interval / 2)
            self.current_time -= self.start_jump
            cmd.time = 0

        self.queue_df = self.queue_df.append({
            'Time': cmd.time,
            'RoundTime': np.floor(cmd.time / self.interval) * self.interval,
            'Command': cmd.cmd_type,
            'Value': cmd.val,
            'X': cmd.x,
            'Y': cmd.y,
        }, ignore_index=True)

    def clean_similar(self):
        self.queue_df.drop_duplicates(subset=['RoundTime', 'Command'], inplace=True)
        self.queue_df.sort_values(by=['Time'], inplace=True)

    @staticmethod
    def draw_cmd_rect(frame, x, y, color, val):
        s = 20
        pt1 = (x - s, y - s)
        pt2 = (x + s, y + s)
        pt_text = (x - s, y + s + 15)
        cv.rectangle(frame, pt1, pt2, color, 2)
        cv.putText(frame, f'{val:0.2f}', pt_text, cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    def show_command(self, frame, cmd: Command):
        HexBot.draw_cmd_rect(frame, int(cmd.x), int(cmd.y), Command.COLOR[cmd.cmd_type], cmd.val)

    def show_queue(self, frame, target_x: float, velocity: float):
        for _, row in self.queue_df.iterrows():
            cmd_type = row['Command']
            cmd_time = row['Time']
            color = Command.COLOR[cmd_type]
            y = int(row['Y'])
            val = row['Value']

            dt = cmd_time - self.current_time
            dx = velocity * dt
            x = int(target_x - dx)

            HexBot.draw_cmd_rect(frame, x, y, color, val)

    def next_command(self) -> CommandType:
        query = f'Time <= {self.current_time}'
        queue_next = self.queue_df.query(query)

        n = queue_next.shape[0]
        if n == 0:
            return None

        cmd_counts = queue_next['Command'].value_counts()
        cmd = cmd_counts.index[0]

        self.queue_df.drop(queue_next.index, inplace=True)
        return cmd

    def execute_command(self, cmd_type: CommandType):
        if cmd_type is None:
            return

        button = Command.BUTTONS[cmd_type]
        action = Command.ACTION_NAME[cmd_type]

        pyautogui.press(button)

        self.exec_df = self.exec_df.append({
            'Time': self.current_time,
            'Command': action,
        }, ignore_index=True)

    def clear_commands(self):
        self.start_jump = 0
        self.current_time = 0
        self.queue_df.drop(self.queue_df.index, inplace=True)
        self.exec_df.drop(self.exec_df.index, inplace=True)

    def save_exec(self, path: str = 'data/exec.csv'):
        self.exec_df.to_csv(path, index=None)
