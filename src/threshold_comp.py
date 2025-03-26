import lip
import cv2 as cv
import numpy as np
def main():
    
    #img = cv.imread('img_data/Rose.jpg', cv.IMREAD_GRAYSCALE)
    #img = cv.imread('img_data/acat.jpg', cv.IMREAD_GRAYSCALE)
    img = cv.imread('img_data/caballo.webp', cv.IMREAD_GRAYSCALE)

    img = cv.resize(img, (600, 400))

    threshold_otsu = lip.OtsuThreshold(img)
    threshold_ridler_calvard = lip.RidlerCalvardThreshold(img)

    # Aplicar umbralización
    _, img_threshold_otsu = cv.threshold(img, threshold_otsu, 255, cv.THRESH_BINARY)
    _, img_threshold_ridler_calvard = cv.threshold(img, threshold_ridler_calvard, 255, cv.THRESH_BINARY)

    lip.show_images_together([img, img_threshold_otsu, img_threshold_ridler_calvard], ['Imagen', 'Imagen con umbral otsu', 'Imagen con umbral ridler calvard'])
   
if __name__ == "__main__":
    main()
