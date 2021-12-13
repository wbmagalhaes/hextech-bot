import cv2 as cv
import pandas as pd

from time import sleep
from datetime import datetime

from PIL import ImageGrab

from utils.timer import timefunc
from utils.command import Command
from utils.image_processing import prepare_image


VELOCITY = -211  # pixel/s
TARGET_X = 420

XMIN_1, XMAX_1 = 550, 610
XMIN_2, XMAX_2 = 1050, 1110
YMIN, YMAX = 100, 800

CAPTURE = cv.VideoCapture('data/video_sample.mp4')
FRAME_LIMIT = 1 / 60

TIME_WINDOW = 0.1
TIME_OFFSET = 0.0
START_TIME = datetime.now()

queue_df = pd.DataFrame(columns=['Time', 'Command'])
exec_df = pd.DataFrame(columns=['Frame', 'Command'])


@timefunc(verbose=False)
def cut_and_process(frame, xmin, xmax, ymin, ymax, current_time):
    cut = frame[ymin:ymax, xmin:xmax, :]

    gray = prepare_image(cut)
    cmd = Command.find(gray, xmin, TARGET_X, VELOCITY, current_time, threshold=0.75)

    if cmd:
        cut = cmd.mark(cut)

    height, width, _ = cut.shape
    cv.rectangle(cut, (0, 0), (width, height), (0, 0, 255), 2)

    return cmd


def add_to_queue(cmd, df):
    if cmd:
        return df.append({
            'Command': cmd.type,
            'Time': round(cmd.time, 1)
        }, ignore_index=True)

    return df


n = 0
while CAPTURE.isOpened():
    # frame = np.array(ImageGrab.grab(bbox=(x, y, x + w * n, y + h), all_screens=True))
    # frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

    ret, frame = CAPTURE.read()
    if not ret:
        print('Stream end. Exiting...')
        break
    n += 1

    # time control
    now = datetime.now()
    current_time = (now - START_TIME).total_seconds()

    # look at frame
    cmd1 = cut_and_process(frame, XMIN_1, XMAX_1, YMIN, YMAX, current_time)
    cmd2 = cut_and_process(frame, XMIN_2, XMAX_2, YMIN, YMAX, current_time)

    # build queue
    queue_df = add_to_queue(cmd1, queue_df)
    queue_df = add_to_queue(cmd2, queue_df)

    queue_df.drop_duplicates(subset=['Time'], inplace=True)
    queue_df.sort_values(by=['Time'], inplace=True)

    # execute queue
    TMIN = current_time - TIME_OFFSET - TIME_WINDOW
    TMAX = current_time - TIME_OFFSET + TIME_WINDOW
    query = f'Time > {TMIN} & Time < {TMAX}'
    queue_next = queue_df.query(query)
    lines = queue_next.shape[0]

    cmd = None
    if lines > 0:
        next_line = queue_next.iloc[0]
        name = next_line.name
        cmd = next_line['Command']

        queue_next.drop(name)
        cv.rectangle(frame, (TARGET_X, YMIN), (TARGET_X, YMAX), Command.COLOR[cmd], 2)

    exec_df = exec_df.append({
        'Frame': n,
        'Command': cmd,
    }, ignore_index=True)

    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

    time_diff = (datetime.now() - now).total_seconds()
    if (time_diff < FRAME_LIMIT):
        sleep(FRAME_LIMIT - time_diff)


CAPTURE.release()
cv.destroyAllWindows()


exec_df.to_csv('data/queue.csv')
print(queue_df)

exec_df.to_csv('data/exec.csv')
print(exec_df)
