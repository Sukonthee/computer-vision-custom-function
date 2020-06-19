import cv2


def increaseBrightness(img, value=30):
    '''
        img: image
        value: increase brightness pixel values 

        return increase brightness image
    '''
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def main():

    # read image
    image = cv2.imread("image/coins.png")

    # call function
    increase_image = increaseBrightness(image, 50)

    cv2.imshow("original", image)
    cv2.imshow("result", increase_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
