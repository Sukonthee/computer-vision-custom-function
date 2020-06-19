import cv2
import numpy as np

height = 500
width = 500


def bboxOffset2Crop(x1, y1, w, h, offset, shiftUpOffset):
    '''
        x1 : top left x
        y1 : top left y
        w : width from x1
        h : height from y1
        offset : increase x overall ofset (0.0-1.0)
        shiftUpOffset : increase (x1, y1) offset (0.0-1.0)

        return upx1, upy1, upx2, uy2
    '''
    upX1 = int(x1 - (w * offset))
    upY1 = int(y1 - (h * offset) - (h * shiftUpOffset))
    upX2 = int(x1 + (w * offset) + w)
    upY2 = int(y1 + (h * offset) + h)
    return upX1, upY1, upX2, upY2


def main():

    # create image
    blank_image = np.zeros((height, width, 3), np.uint8)
    x1 = 100
    y1 = 100
    w = 100
    h = 100

    # draw original offset
    cv2.rectangle(blank_image, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 3)

    # call function
    (upx1, upy1, upx2, upy2) = bboxOffset2Crop(x1, y1, w, h, 0.3, 0.0)

    # draw increase offset
    cv2.rectangle(blank_image, (upx1, upy1), (upx2, upy2), (0, 0, 255), 3)
    cv2.imshow("origninal", blank_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
