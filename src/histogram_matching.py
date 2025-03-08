import numpy as np
import cv2
import ena

def main():
    img1 = cv2.imread('img_data/Y5.jpg')    
    img2 = cv2.imread('img_data/Y4.jpg')

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img_result = ena.HistogramMatchingGray(img1, img2)

    ena.show_images_together([img2, img_result], ['Original', 'Result'])
    
    img2 = cv2.imread('img_data/lena.jpg')   
    img2 = cv2.resize(img2, (800, 800))

    img1 = cv2.imread('img_data/sun.jpg')
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

    img_result_rgb = ena.HistogramMatchingRGB(img1, img2)

    ena.show_images_together([img2, img_result_rgb], ['Original', 'Result'])

if __name__ == '__main__':
    main()
