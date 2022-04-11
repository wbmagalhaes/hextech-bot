from re import T
import cv2 as cv

from utils.hexbot import HexBot

# região do jogo a analisado pelo bot
XMIN, XMAX = 280, 350
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

count = 0
while True:
    count += 1

    frame = cv.imread(f'./data/frames/frame{count}.png')
    if frame is not None:
        print('Frame', count)

        cmd = BOT.process_frame(frame, TARGET_X, VELOCITY, verbose=True)

        if (cmd):
            BOT.show_command(frame, cmd)

        cv.imshow(f'HextechBot - {count}', frame)

        key = cv.waitKey()
        if key == ord('p'):
            count = max(count - 2, 0)
        elif key == ord('q'):
            cv.destroyAllWindows()
            break
