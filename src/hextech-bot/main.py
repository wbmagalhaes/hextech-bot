import cv2 as cv

from time import perf_counter

from utils.hexbot import HexBot
from utils.command import Command
from utils.window_capture import WindowCapture

XMIN, XMAX = 280, 330
YMIN, YMAX = 10, 550
REGION = (XMIN, XMAX, YMIN, YMAX)

METHOD = cv.TM_CCOEFF_NORMED
THRESHOLD = 0.60

TARGET_X = 155
VELOCITY = -170  # pixel/s

BOT = HexBot(REGION, METHOD, THRESHOLD)

# WindowCapture.list_window_names()
wincap = WindowCapture('HextechMayhem')

frame_count = 0
prev_time = perf_counter()


while True:
    # time control
    frame_count += 1
    now = perf_counter()
    BOT.set_current_time(now)

    # capture
    frame = wincap.get_screenshot()

    # frame processing
    BOT.process_frame(frame, TARGET_X, VELOCITY)
    BOT.show_queue(frame, TARGET_X, VELOCITY)

    # get next
    cmd = BOT.next_command()

    if cmd:
        cv.rectangle(frame, (TARGET_X, YMIN), (TARGET_X, YMAX), Command.COLOR[cmd], 3)
        # cv.imwrite(f'data/frames/frame{frame_count}_{cmd}.png', frame)

    # execute
    BOT.execute_command(cmd)

    # fps counter
    fps = 1 / (now - prev_time)

    cv.putText(frame, f'FPS: {fps:0.0f}', (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    prev_time = now

    # show
    cv.imshow('HextechBot', frame)

    # wait key 'Q' to exit
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        BOT.save_exec('data/exec_1-1 (2).csv')
        break
