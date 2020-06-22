import math
import cv2
import numpy as np
import imutils


class AutoRotate():
    '''
    ฟังก์ชั่นนี้สามารถใช้ในการปรับมุมภาพ, คำนวณมุมการหมุนของวัตถุในภาพ เพื่อนำใช้ในการหมุน
    นอกจากนี้ยังมีฟังก์ชันค้นหา 4 จุดคู่อันดับของมุมสี่เหลี่ยมเพื่อใช้ในการแสดงภาพเฉพาะหรือหมุนภาพเฉพาะส่วน
    ซึ่งฟังก์ชันนี้ประกอบด้วยฟังก์ชันย่อย 4 ฟังก์ชัน ได้แก่ find_angle, rotation, order_points และ point_tranfrom
    '''
    def __init__(self):
        self.__area = 100

    def find_angle(self, image, increase_size, lower_threshold,
                   upper_threshold):
        '''
        ฟังก์ชันนี้มีวัตถุประสงค์เพื่อคำนวณหามุมของวัตถุ ด้วยการค้นหาตำแหน่งเฉลี่ยที่เป็นไปได้ของเส้นตรงภายในรูปภาพมาใช้ในสมการทางคณิตศาสตร์
        เพื่อคำนวณมุมการหมุนที่ทำให้ตำแหน่งของเส้นตรงที่นำมาใช้หมุนภาพนั้นอยู่ในระนาบแนวนอน(แกน X) ซึ่งค่ามุมที่ได้รับนั้นเป็นได้ทั้งบวกและลบ 

        *ข้อแนะนำ* ไม่ควรนำไปใช้กับภาพตัวอักษรเพียงตัวเดียว 

        :param image: ภาพสี RGB ที่ผู้ใช้ต้องการ
        :param increase_size: จำนวนครั้งของการขยายที่ถูกนำไปใช้เพิ่มความหนาให้วัตถุ
        :type increase_size: integer
        :param lower_threshold: ขีดจำกัดต่ำสุดสำหรับกระบวนการลบภาพพื้นหลัง มีค่าตั้งแต่ 0-255
        :type lower_threshold: integer 
        :param upper_threshold: ขีดจำกัดสูงสุดสำหรับกระบวนการลบภาพพื้นหลัง มีค่าตั้งแต่ 0-255
        :type upper_threshold: integer 
        :return: 
            - image: ภาพปรากฎเส้นตรงที่ถูกนำไปใช้ในการคำนวณหามุม
            - angle(float): ค่ามุมที่ได้จากการคำนวณ ซึ่งเป็นได้ทั้งค่าบวกและค่าลบ
        '''
        binary_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # เตรียมภาพเพื่อทำการหาตำแหน่งค่าเฉลี่ยนของเส้นตรงที่เป็นไปได้
        # Prepare an image to find the average position of the straight lines as possible.
        edges = cv2.Canny(binary_image, lower_threshold, upper_threshold)
        kernel = np.ones((3, 3), np.uint8)
        img_closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        img_dilation = cv2.dilate(img_closing,
                                  kernel,
                                  iterations=int(increase_size))

        # หาเส้นตรงแลำคำนวณหามุมที่จะหมุมของเส้นตรงที่เจอ
        # Find a straight line and calculate the angle that will be the angle of the straight line found
        angles = []
        lines = cv2.HoughLinesP(img_dilation,
                                1,
                                math.pi / 180.0,
                                upper_threshold,
                                minLineLength=100,
                                maxLineGap=10)

        if lines is not None:
            for x1, y1, x2, y2 in lines[0]:

                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                angles.append(angle)
                angle = np.median(angles)

                img_line = cv2.line(image.copy(), (x1, y1), (x2, y2),
                                    (0, 255, 0), 2)

        elif lines is None:
            img_line = image
            angle = 0
            print("Not find line in image")

        # ส่งคืนรูปภาพมีเส้นตรงที่ใช้ในการคำนวณและมุม
        # return image has straight lines used in the calculation and angle
        return img_line, angle

    def rotation(self, image, angle):
        '''
        ฟังก์ชั่นนี้มีวัตถุประสงค์เพื่อหมุนภาพโดยรับมุมจากผู้ใช้ หรือมุมที่ได้จากฟังก์ชัน find_angle ซึ่งมุมบวกและลบของค่ามุมจะมีผลต่อการหมุนของภาพ
        ตามแบบการหมุนของวงกลมหน่วย หรือก็คือ ถ้าหากมุมการหมุนเป็นบวกจะทำให้ภาพหมุนทวนเข็มนาฬิกา แต่ถ้ามุมการหมุนเป็นลบจะทำให้ภาพหมุนตามเข็มนาฬิกา
        โดยฟังก์ชั่นนี้ใช้การหมุนที่ไม่ครอบตัดภาพ
        
        *ข้อแนะนำ* หากผู้ใช้ใช้ค่ามุมที่ได้จากฟังก์ชัน find_angle และภาพที่เข้ามาเป็นภาพที่ต้องพิจารณาข้อมูลในแนวตั้ง ผู้ใช้จำเป็นต้องบวกมุม 90 องศา เพื่อให้ภาพหมุนมาอยู่ในแนวตั้ง

        :param image: ภาพสี RGB ที่ผู้ใช้ต้องการ
        :param angle: มุมที่ใช้ในการหมุน
        :type angle: float
        :return: 
            - image: ภาพที่ทำการหมุนเรียบร้อยแล้ว
        '''

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        # คำนวณขนาดภาพตามการหมุนเพื่อป้องกันไม่ให้ภาพบาง่วนถูกครอบตัด
        # calculate image size based on rotation to prevent some images from being cropped
        scale = 1.0
        M = cv2.getRotationMatrix2D(center, angle, scale)

        newX, newY = w * scale, h * scale
        r = np.deg2rad(angle)
        newX, newY = (abs(np.sin(r) * newY) + abs(np.cos(r) * newX),
                      abs(np.sin(r) * newX) + abs(np.cos(r) * newY))

        (tx, ty) = ((newX - w) / 2, (newY - h) / 2)
        M[0, 2] += tx
        M[1, 2] += ty

        # คำสั่งที่ใช้ในการหมุนภาพ
        # commands used to rotate images.
        rotated = cv2.warpAffine(image, M, (int(newX), int(newY)))

        # คืนภาพที่ทำการหมุนเรียบร้อยแล้ว
        # Return the image that has been successfully rotated.
        return rotated

    def order_points(self,
                     image,
                     lower_threshold,
                     upper_threshold,
                     area=None,
                     mode='mode'):
        '''
        ฟังก์ชั่นนี้ใช้ในการค้นหาจุดคู่อันดับ(x,y) 4 จุดของวัตถุจากการวาดสี่เหลี่ยมล้อมรอบวัตถุ โดยมีโหมดการทำงาน 2 โหมดซึ่งทั้งสองโหมดจะค้นหาจุดคู่อันดับ
        ประกอบด้วย จุดคู่อันดับด้านซ้ายบน, จุดคู่อันดับด้านขวาบน, จุดคู่อันดับด้านล่างขวาและจุดคู่อันดับด้านซ้ายล่างตามลำดับ ด้วยการใช้การวาดกรอบ contour

        :param image: ภาพสี RGB ที่ผู้ใช้ต้องการ
        :param lower_threshold: ขีดจำกัดต่ำสุดสำหรับกระบวนการลบภาพพื้นหลัง มีค่าตั้งแต่ 0-255
        :type lower_threshold: integer 
        :param upper_threshold: ขีดจำกัดสูงสุดสำหรับกระบวนการลบภาพพื้นหลัง มีค่าตั้งแต่ 0-255
        :type upper_threshold: integer
        :param area: พื้นที่ขั้นต่ำของค่าสีที่ผู้ใช้ต้องการ หากพื้นที่ของค่าสีที่เจอมีค่าน้อยกว่าจะไม่ถูกนำมาพิจารณา (pixel)
        :type area: integer or None 
        :param mode:
            1. mode = 'points' 
                โหมดนี้จะวาดกรอบล้อมรอบวัตถุตามรูปร่างของวัตถุ หากวัตถุนั้นเป็นวัตถุที่มีรูปร่างเป็นสี่เหลี่ยม
                จะส่งจุดคู่อันดับ 4 จุดที่ได้คือ จุดคู่อันดับด้านซ้ายบน, จุดคู่อันดับด้านขวาบน, จุดคู่อันดับด้านล่างขวาและจุดคู่อันดับด้านซ้ายล่างตามลำดับ
            2. mode = 'boxes' 
                โหมดนี้จะวาดกรอบสี่เหลี่ยมด้านเท่าล้อมรอบวัตถุและทำการส่งจุดคู่อันดับที่ 4 จุด คือ จุดคู่อันดับด้านซ้ายบน, จุดคู่อันดับด้านขวาบน, 
                จุดคู่อันดับด้านล่างขวาและจุดคู่อันดับด้านซ้ายล่างตามลำดับ โดยที่ไม่สนว่าวัตถุนั้นมีรูปร่างเป็นสี่เหลี่ยมหรือไม่
        :return: 
            - image: ภาพที่มีการมาร์คจุดคู่อันดับ
            - boxes[list(float)]: จุดคู่อันดับทั้ง 4 จุดทั้งหมดที่เจอ
            
        Example 
            boxes = ([[tlX,tlY], [trX,trY], [brX,brY], [blX,blY]], ...)
        '''
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (T, threshInv) = cv2.threshold(gray, lower_threshold, upper_threshold,
                                       cv2.THRESH_BINARY_INV)

        # หาจุดตำแหน่งของวัตถุในภาพ
        cnts = cv2.findContours(threshInv.copy(), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        new_img = image.copy()
        mode = str(mode)
        boxes = []
        i = 0

        for c in cnts:
            area = self.__area if area is None else area
            if cv2.contourArea(c) > int(area):

                # เลือกโหมดการทำงาน
                # select mode
                if mode is 'points':

                    # รูปร่างของ contour
                    # approximate the contour
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

                    # กำหนดรูปร่างโดยประมาณที่มีสี่จุด
                    # Set approximated contour has four points
                    if len(approx) == 4:
                        box = np.concatenate(
                            (approx[1], approx[0], approx[3], approx[2]))
                        cv2.drawContours(new_img, [box], -1, (0, 255, 0), 2)

                        i = i + 1
                        # กำหนดจุด tl, tr, br, bl
                        # Set points tl, tr, br, bl
                        rect = np.zeros((4, 2), dtype="float32")
                        s = box.sum(axis=1)
                        rect[0] = box[np.argmin(s)]
                        rect[2] = box[np.argmax(s)]

                        diff = np.diff(box, axis=1)
                        rect[1] = box[np.argmin(diff)]
                        rect[3] = box[np.argmax(diff)]

                        print("Object #{}:".format(i))
                        print(rect)

                        # loop over the original points and draw
                        for (x, y) in rect:
                            cv2.circle(new_img, (int(x), int(y)), 2,
                                       (0, 255, 0), -1)
                            cv2.putText(
                                new_img, "#{}".format(i),
                                (int(rect[0][0] - 15), int(rect[0][1] - 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255),
                                2)
                    else:
                        continue

                elif mode is 'boxes':
                    i = i + 1

                    # วาด contour แบบสี่เหลี่ยมด้านเท่า
                    # Draw a square-shaped contour
                    box = cv2.minAreaRect(c)
                    box = cv2.cv.BoxPoints(
                        box) if imutils.is_cv2() else cv2.boxPoints(box)
                    box = np.array(box, dtype="int")
                    cv2.drawContours(new_img, [box], -1, (0, 255, 0), 2)

                    # กำหนดจุด tl, tr, br, bl
                    # Set points tl, tr, br, bl
                    rect = np.zeros((4, 2), dtype="float32")
                    s = box.sum(axis=1)
                    rect[0] = box[np.argmin(s)]
                    rect[2] = box[np.argmax(s)]

                    diff = np.diff(box, axis=1)
                    rect[1] = box[np.argmin(diff)]
                    rect[3] = box[np.argmax(diff)]

                    print("Object #{}:".format(i))
                    print(rect)

                    # loop over the original points and draw
                    for (x, y) in box:
                        cv2.circle(new_img, (int(x), int(y)), 2, (0, 255, 0),
                                   -1)
                        cv2.putText(new_img, "#{}".format(i),
                                    (int(rect[0][0]), int(rect[0][1])),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                    (0, 255, 255), 2)
            else:
                continue
            boxes.append(rect)
        # return image with mark points and array of 4 corners.
        return new_img, boxes

    def point_transform(self, image, four_point):
        '''
        ฟังก์ชั่นนี้ใช้ในการเปลี่ยนมุมมองของภาพหรือทำการหมุนภาพเฉพาะจุดที่ผู้ใช้ต้องการ ซึ่งจะใช้ชุดสี่มุมของวัตถุซึ่งประกอบด้วยมุมบนซ้ายมุมขวาบนมุมขวาล่าง
        และมุมล่างซ้ายตามลำดับที่ได้รับจากผู้ใช้เพียงชุดเดียวเพื่อใช้ในการเปลี่ยนมุมของภาพ
        
        :param image: ภาพสี RGB ที่ผู้ใช้ต้องการ
        :param four_points: จุดคู่อันดับทั้ง 4 จุดรอบวัตถุประกอบด้วย จุดคู่อันดับด้านซ้ายบน, จุดคู่อันดับด้านขวาบน, จุดคู่อันดับด้านล่างขวาและจุดคู่อันดับด้านซ้ายล่างตามลำดับ
        :type four_points: list 
        :return: 
            - image: ภาพเฉพาะจุดที่ทำการหมุนเรียบร้อยแล้ว
        '''
        (tl, tr, br, bl) = four_point

        # คำนวณขนาดความกว้างแลพความสูงของภาพใหม่
        # Recalculate the size, width and height of the image.
        widthA = np.sqrt(((br[0] - bl[0])**2) + ((br[1] - bl[1])**2))
        widthB = np.sqrt(((tr[0] - tl[0])**2) + ((tr[1] - tl[1])**2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0])**2) + ((tr[1] - br[1])**2))
        heightB = np.sqrt(((tl[0] - bl[0])**2) + ((tl[1] - bl[1])**2))
        maxHeight = max(int(heightA), int(heightB))

        # สร้างจุดปลายทางเพื่อรับ "birds eye view"
        # construct the set of destination points to obtain a "birds eye view"
        dst = np.array([[0, 0], [maxWidth - 1, 0],
                        [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
                       dtype="float32")

        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(four_point.astype(np.float32), dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        # return the warped image
        return warped


def main():
    image = cv2.imread("image/rotate.jpg")
    image = imutils.resize(image, width=300)
    r = AutoRotate()
    line, angle = r.find_angle(image, 3, 100, 200)
    rotation_image = r.rotation(image, angle)
    cv2.imshow("line", line)
    cv2.imshow("original", image)
    cv2.imshow('rotated', rotation_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
