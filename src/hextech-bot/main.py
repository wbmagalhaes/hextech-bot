import cv2 as cv

from time import perf_counter

from utils.hexbot import HexBot
from utils.window_capture import WindowCapture

XMIN, XMAX = 225, 275
YMIN, YMAX = 50, 550
REGION = (XMIN, XMAX, YMIN, YMAX)

METHOD = cv.TM_CCOEFF_NORMED
THRESHOLD = 0.6

TARGET_X = 135
VELOCITY = -170  # pixel/s
TIME_WINDOW = 0.1

BOT = HexBot(REGION, METHOD)

# WindowCapture.list_window_names()
wincap = WindowCapture('HextechMayhem')

START_TIME = perf_counter()

frame_count = 0
prev_time = perf_counter()


while True:
    # time control
    now = perf_counter()
    current_time = now - START_TIME
    frame_count += 1

    # capture
    frame = wincap.get_screenshot()

    # frame processing
    BOT.process_frame(frame, frame_count, current_time, THRESHOLD, TARGET_X, VELOCITY)
    BOT.show_queue(frame, current_time, TARGET_X, VELOCITY)

    # excute
    tmax = current_time + TIME_WINDOW
    cmd = BOT.next_command(current_time, tmax)
    BOT.execute_command(current_time, frame_count, cmd)

    # fps counter
    fps = 1 / (now - prev_time)
    cv.putText(frame, f'FPS: {fps:0.0f}', (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    prev_time = now

    # show
    cv.imshow('HextechBot', frame)

    # wait key 'Q' to exit
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        BOT.save_exec()
        BOT.save_queue()
        break
