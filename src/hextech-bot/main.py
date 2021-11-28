import cv2 as cv
import numpy as np

from PIL import ImageGrab

from find_command import find_command, mark_command

x, y = 2665, 100
w, h = 70, 760


while True:
    frame = np.array(ImageGrab.grab(bbox=(x, y, x + w, y + h), all_screens=True))
    img = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    cmd = find_command(img_gray, threshold=0.2)

    if cmd:
        img = mark_command(img, cmd)
        print(cmd.type)

    cv.imshow("frame", img)

    if cv.waitKey(1) & 0Xff == ord('q'):
        break
