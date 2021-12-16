import cv2 as cv
import numpy as np
import pandas as pd

import pyautogui

from utils.timer import timefunc
from utils.image_processing import prepare_image

from utils.command import Command


class HexBot:

    def __init__(self, region, method):
        self.xmin, self.xmax, self.ymin, self.ymax = region
        self.method = method

        self.queue_df = pd.DataFrame(columns=['Time', 'Command'])
        self.exec_df = pd.DataFrame(columns=['Frame', 'Command'])

    @timefunc(verbose=False)
    def process_frame(self, frame, frame_count, time, threshold, target_x, velocity):

        cut = frame[self.ymin:self.ymax, self.xmin:self.xmax, :]
        gray = prepare_image(cut)

        cmd = self.find_cmd(gray, threshold, target_x, velocity, time)
        self.add_to_queue(frame_count, cmd)

        self.queue_df.drop_duplicates(subset=['Time', 'Command'], inplace=True)
        self.queue_df.sort_values(by=['Time'], inplace=True)

        height, width, _ = cut.shape
        cv.rectangle(cut, (0, 0), (width, height), (0, 0, 255), 2)

    def find_cmd(self, img, threshold, target_x, velocity, time) -> Command:

        results_val = []
        results = []

        for type in Command.TYPES:
            template = Command.TEMPLATES[type]
            mask = Command.MASK[type]

            val, top_left, bottom_right = self.find_template(img, template, mask)
            cmd = Command(type, val, (self.xmin, self.ymin), top_left, bottom_right, target_x, velocity, time)

            results.append(cmd)
            results_val.append(val)

        max_ind = np.argmax(results_val)
        result = results[max_ind]

        if result.val > threshold:
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

    def add_to_queue(self, frame_count, cmd: Command):
        if cmd:
            self.queue_df = self.queue_df.append({
                'Frame': frame_count,
                'Time': cmd.time,
                'Command': cmd.cmd_type,
                'Value': cmd.val,
                'Y': cmd.y
            }, ignore_index=True)

    def show_queue(self, frame, time, target_x, velocity):
        for _, row in self.queue_df.iterrows():

            cmd = row['Command']
            cmd_time = row['Time']
            y = int(row['Y'])
            val = row['Value']

            dt = cmd_time - time
            dx = velocity * dt
            x = int(target_x - dx)

            s = 20
            pt1 = (x - s, y - s)
            pt2 = (x + s, y + s)

            color = Command.COLOR[cmd]
            cv.rectangle(frame, pt1, pt2, color, 2)
            cv.putText(frame, f'{val:0.2f}', pt1, cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)

    def next_command(self, tmin, tmax) -> Command:
        query = f'Time >= {tmin} & Time <= {tmax}'
        queue_next = self.queue_df.query(query)

        if queue_next.shape[0] == 0:
            return None

        cmd_counts = queue_next["Command"].value_counts()
        cmd = cmd_counts.index[0]

        self.queue_df.drop(queue_next.index, inplace=True)

        return cmd

    def execute_command(self, time, n_frame, cmd):
        if cmd is None:
            return

        button = Command.BUTTONS[cmd]
        pyautogui.press(button)

        self.exec_df = self.exec_df.append({
            'Frame': n_frame,
            'Time': time,
            'Command': Command.ACTION_NAME[cmd],
        }, ignore_index=True)

    def save_queue(self):
        self.queue_df.to_csv('data/queue.csv')

    def save_exec(self):
        self.exec_df.to_csv('data/exec.csv')
