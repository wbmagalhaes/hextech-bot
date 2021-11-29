import cv2 as cv
import numpy as np

from PIL import ImageGrab

from find_command import find_command, mark_command
from utils.image_processing import prepare_image


x, y = 2420, 100
w, h = 61, 760
offset = 0
# dt = 700 ms

while True:
    frame = np.array(ImageGrab.grab(bbox=(x, y, x + w * 10, y + h), all_screens=True))
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

    # frame = cv.imread('frame.png')

    height, width, _ = frame.shape
    n = (width - offset) // w

    cuts = []
    for i in range(n):
        cut = frame[:, w * i + offset: w * (i + 1) + offset, :]

        img_gray = prepare_image(cut)
        cmd = find_command(img_gray, threshold=0.75)

        if cmd:
            cut = mark_command(cut, cmd)

        height, width, _ = cut.shape
        cv.rectangle(cut, (0, 0), (width, height), (0, 0, 255), 2)

        cuts.append(cut)

    frame = np.hstack(cuts)
    cv.imshow("frame", frame)

    if cv.waitKey(1) & 0Xff == ord('q'):
        break
