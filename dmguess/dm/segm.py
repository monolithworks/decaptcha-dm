import cv2
import numpy as np
import sys

class Segmenter(object):

    def __init__(self):
        pass

    def __findBox(self, image):
        #Binarize it to make the contour algorithm work fine
        thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)[1]
        _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE,
                                  cv2.CHAIN_APPROX_SIMPLE)

        #Its a hierarchy, the last contour is the BOX
        box = self.__findCountours(cnts, self.__filterBoxContour)[-1]
        x,y,w,h = map(int,box)
        crop_img = image[y+2:y+h-2, x+2:x+w-2]
        return crop_img


    def __filterBoxContour(self, bb):
        brArea = bb[2]*bb[3]
        #print(brArea)
        return (brArea>2400)  and (brArea<6000)

    def __filterDigitContour(self, bb):
        brArea = bb[2]*bb[3]
        return brArea>100

    def __findDigits(self, box_img):
        thresh = cv2.threshold(box_img, 127, 255, cv2.THRESH_BINARY_INV)[1]
        _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)

        return self.__findCountours(cnts, self.__filterDigitContour)

    def __findCountours(self, cnts, filter):
        boundRects = []
        for c in cnts:
            approx = cv2.approxPolyDP(c, 1, True)
            boundRect = cv2.boundingRect(approx)
            if filter(boundRect):
                #print('BB detected')
                boundRects.append(boundRect)
        return boundRects

    def __filterBoundingBoxes(self, boxes):
        #We sort them in the X axis
        boxes = sorted(boxes,key=lambda x: x[0])
        if len(boxes) != 4:
            print('Warning, detected digts != 4', len(boxes))

        #TODO add more code here
        return boxes

    def __getDigits(self,img,boxes):
        digits = []
        boxes = self.__filterBoundingBoxes(boxes)
        for i,boundRect in enumerate(boxes):
            x,y,w,h = map(int,boundRect)
            crop_img = img[y:y+h, x:x+w]
            digits.append(crop_img)
        return digits

    def segment(self, image):
        #Get the box with the 4 digits
        box_img = self.__findBox(image)
        #cv2.imshow("Image", box_img)
        #cv2.waitKey(0)

        #find the digits in the cropped box
        boxes=self.__findDigits(box_img)
        return self.__getDigits(box_img,boxes)

if __name__ == '__main__':

    image     = cv2.imread(sys.argv[1])
    segmenter = Segmenter()

    digits=segmenter.segment(image)
    for i,digit in enumerate(digits):
        cv2.imshow("Digit %d"%i, digit)
        #cv2.imwrite("PATH",digit)
    cv2.waitKey(0)
