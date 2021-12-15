from utils.coord import Coord


class Square():
    def __init__(self, xmin, xmax, ymin, ymax):
        self.top_left = Coord(xmin, ymin)
        self.bottom_right = Coord(xmax, ymax)

    def cut_frame(self, frame):
        xmin = self.top_left.x
        ymin = self.top_left.y

        xmax = self.bottom_right.x
        ymax = self.bottom_right.y

        return frame[ymin:ymax, xmin:xmax, :]
