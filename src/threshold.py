import ena
import cv2 as cv
import numpy as np
def main():
    
    img = cv.imread('img_data/Rose.jpg', cv.IMREAD_GRAYSCALE)

    img = cv.resize(img, (800, 600))

    threshold = ena.RidlerCalvardThreshold(img)

    # Aplicar umbralización
    _, img_threshold = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)

    cv.imshow('Imagen', img)
    cv.imshow('Imagen con umbral', img_threshold)
    cv.waitKey(0)
    cv.destroyAllWindows()
   
if __name__ == "__main__":
    main()
