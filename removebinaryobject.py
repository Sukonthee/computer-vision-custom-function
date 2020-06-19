import cv2
import numpy as np


def bwareaopenCV2(image, minSize):
    '''
        image: bianry image
        minSize: minsize object don't want to remove

        return existing object image
    '''
    nbComponents, output, stats, centroids = cv2.connectedComponentsWithStats(
        image, connectivity=8)
    sizes = stats[1:, -1]
    nbComponents = nbComponents - 1
    result = np.zeros((output.shape))
    for i in range(0, nbComponents):
        if sizes[i] >= minSize:
            result[output == i + 1] = 255
    return result


def main():

    # read image
    image = cv2.imread("image/coins.png")

    # convert image to binary
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # call removebinaryobject function
    result = bwareaopenCV2(thresh1, 1000)

    cv2.imshow("original", thresh1)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
