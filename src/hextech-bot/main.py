import cv2 as cv

from time import sleep
from datetime import datetime

from PIL import ImageGrab

from utils.command import Command
from utils.hexbot import HexBot


XMIN_1, XMAX_1 = 550, 610
XMIN_2, XMAX_2 = 1050, 1110
YMIN, YMAX = 100, 800

METHOD = cv.TM_CCOEFF_NORMED
THRESHOLD = 0.75

TARGET_X = 420
VELOCITY = -211  # pixel/s

TIME_WINDOW = 0.1
TIME_OFFSET = 0.0

SQUARES = [
    (XMIN_1, XMAX_1, YMIN, YMAX),
    (XMIN_2, XMAX_2, YMIN, YMAX),
]

BOT = HexBot(SQUARES, METHOD, THRESHOLD, TARGET_X, VELOCITY, TIME_WINDOW, TIME_OFFSET)

START_TIME = datetime.now()
CAPTURE = cv.VideoCapture('data/video_sample.mp4')
FRAME_LIMIT = 1 / 60

frame_count = 0

while CAPTURE.isOpened():
    # frame = np.array(ImageGrab.grab(bbox=(x, y, x + w * n, y + h), all_screens=True))
    # frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

    ret, frame = CAPTURE.read()
    if not ret:
        print('Stream end. Exiting...')
        break

    frame_count += 1

    now = datetime.now()
    current_time = (now - START_TIME).total_seconds()

    BOT.process_frame(frame, current_time)

    cmd = BOT.next_command(current_time)

    if cmd:
        BOT.execute_command(frame_count, cmd)
        cv.rectangle(frame, (TARGET_X, YMIN), (TARGET_X, YMAX), Command.COLOR[cmd], 2)

    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

    time_diff = (datetime.now() - now).total_seconds()
    if (time_diff < FRAME_LIMIT):
        sleep(FRAME_LIMIT - time_diff)


CAPTURE.release()
cv.destroyAllWindows()

BOT.save_queue()
BOT.save_exec()
