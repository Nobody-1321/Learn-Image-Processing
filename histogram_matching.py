import numpy as np
import cv2
import ena

def main():
    img1 = cv2.imread('Y4.jpg')    
    img2 = cv2.imread('Y5.jpg')

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img_result = ena.histogram_matching(img1, img2)

    cv2.imshow('Result', img_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
