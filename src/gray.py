import cv2 as cv
import numpy as np
import ena


def main():
    path =  ena.parse_args_path()
    
    img1 = cv.imread(path)
    img1 = cv.resize(img1, (600, 600), interpolation=cv.INTER_AREA)
    
    img3 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    cv.imshow('Gray_Opencv', img3)

    img2 = ena.bgr_to_gray(img1)
    cv.imshow('Result_Ena', img2)

    
    cv.waitKey(0)    



if __name__ == '__main__':
    main()