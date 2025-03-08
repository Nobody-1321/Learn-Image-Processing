import cv2 as cv
import ena


def main():
    
    img1 = cv.imread("img_data/astro.jpg")    
    img1 = cv.resize(img1, (600, 600), interpolation=cv.INTER_AREA)
    
    img2_cv = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2_ena = ena.BgrToGray(img1)

    ena.show_images_together([img2_cv, img2_ena],["opencv", "ena"])

if __name__ == '__main__':
    main()