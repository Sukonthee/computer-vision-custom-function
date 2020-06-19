import cv2
import numpy as np


def bwareaopenCV2(image, minSize):
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

    cv2.imshow("display", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
