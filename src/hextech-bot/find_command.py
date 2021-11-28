import cv2 as cv
import numpy as np

from utils.timer import timefunc
from utils.command import Command


def find_template(img, template, mask, method=cv.TM_CCOEFF_NORMED):
    img = img.copy()

    # Apply template Matching
    res = cv.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        val = min_val
        top_left = min_loc
    else:
        val = max_val
        top_left = max_loc

    h, w = template.shape

    bottom_right = (top_left[0] + w, top_left[1] + h)
    return val, top_left, bottom_right


@timefunc(verbose=False)
def find_command(img, threshold=0.8):

    results_val = []
    results = []

    for type in Command.TYPES:

        template = Command.TEMPLATES[type]
        mask = Command.MASKS[type]

        val, top_left, bottom_right = find_template(img, template, mask)
        results.append(Command(type, val, top_left, bottom_right))
        results_val.append(val)

    max_ind = np.argmax(results_val)
    result = results[max_ind]

    if result.val > threshold:
        return result

    return None


def mark_command(img, command: Command):
    color = command.getColor()
    cv.rectangle(img, command.top_left, command.bottom_right, color, 2)
    cv.putText(img, f'{command.val:0.2f}', command.top_left, cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return img
