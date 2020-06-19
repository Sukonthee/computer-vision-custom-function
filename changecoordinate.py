import cv2
import numpy as np


def convertCoordinate(frameW, frameH, cloneW, cloneH, x1, y1, x2, y2):
    '''
        frameW: bigger width image
        framwH: bigger height image
        cloneW: smaller width image
        cloneH: smaller height image
        x1, y1, x2, y2: smaller coordinate

        return convert coordinate
    '''
    startX = int((frameW * x1) / cloneW)
    startY = int((frameH * y1) / cloneH)
    width = int((frameW * (x2 - x1)) / cloneW)
    height = int((frameH * (y2 - y1)) / cloneH)
    endX = startX + width
    endY = startY + height
    return startX, startY, endX, endY


def main():

    # create image
    blank_image_1 = np.zeros((300, 300, 3), np.uint8)
    (image1H, image1W) = blank_image_1.shape[:2]
    blank_image_2 = np.zeros((500, 500, 3), np.uint8)
    (image2H, image2W) = blank_image_2.shape[:2]

    x1 = 100
    y1 = 100
    w = 100
    h = 100
    x2 = x1 + w
    y2 = y1 + h

    # call function
    (convertX1, convertY1, convertX2,
     convertY2) = convertCoordinate(image2W, image2H, image1W, image1H, x1, y1,
                                    x2, y2)

    # convert coordinate from small image to bigger image
    cv2.rectangle(blank_image_1, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.rectangle(blank_image_2, (convertX1, convertY1),
                  (convertX2, convertY2), (0, 0, 255), 3)

    # result when convert coordinate
    print(convertX1, convertY1, convertX2, convertY2)

    cv2.imshow("origninal", blank_image_1)
    cv2.imshow("convert", blank_image_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
