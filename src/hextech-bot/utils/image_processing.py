import cv2 as cv


def prepare_image(img):
    # img_gray = img.copy()
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # img_gray = cv.Canny(img, 0, 255)
    return img_gray
