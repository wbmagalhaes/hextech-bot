import cv2 as cv

from time import perf_counter

from utils.hexbot import HexBot
from utils.command import Command
from utils.window_capture import WindowCapture

# região do jogo a analisado pelo bot
XMIN, XMAX = 280, 330
YMIN, YMAX = 10, 550
REGION = (XMIN, XMAX, YMIN, YMAX)

# método de busca por template
METHOD = cv.TM_CCOEFF_NORMED
# match mínimo para considerar uma ação
THRESHOLD = 0.85
# intervalo mínimo entre as ações
INTERVAL = 0.25

# posição do player
TARGET_X = 155
# velocidade das ações em pixel/s
VELOCITY = -170

# o bot
BOT = HexBot(REGION, METHOD, THRESHOLD, INTERVAL)

# WindowCapture.list_window_names()
wincap = WindowCapture('HextechMayhem')

frame_count = 0
prev_time = perf_counter()
started = False

while True:
    # capture
    frame = wincap.get_screenshot()

    if started:
        # time control
        frame_count += 1
        now = perf_counter()
        BOT.set_current_time(now)

        # save frame
        # cv.imwrite(f'data/frames/frame{frame_count}.png', frame)

        # frame processing
        BOT.process_frame(frame, TARGET_X, VELOCITY)
        BOT.show_queue(frame, TARGET_X, VELOCITY)

        # get next
        cmd = BOT.next_command()

        if cmd:
            cv.rectangle(frame, (TARGET_X, YMIN), (TARGET_X, YMAX), Command.COLOR[cmd], 3)
            # cv.imwrite(f'data/frames/frame{frame_count}_{Command.ACTION_NAME[cmd]}.png', frame)

        # execute
        BOT.execute_command(cmd)

        # fps counter
        fps = 1 / (now - prev_time)

        cv.putText(frame, f'FPS: {fps:0.0f}', (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        prev_time = now

    # show
    cv.imshow('HextechBot', frame)

    # wait key 'E' to start
    key = cv.waitKey(1)
    if key == ord('e'):
        started = True

        BOT.clear_commands()
        frame_count = 0

    # wait key 'Q' to exit
    elif key == ord('q'):
        cv.destroyAllWindows()
        BOT.save_exec('data/2-2.csv')
        break
