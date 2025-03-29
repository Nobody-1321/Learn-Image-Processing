import lip
import cv2 as cv
import numpy as np
def main():
    
    #img = cv.imread('img_data/caballo.webp', cv.IMREAD_GRAYSCALE)
    img = cv.imread('img_data/img2.jpeg', cv.IMREAD_GRAYSCALE)
    #img = cv.imread('img_data/pieces.jpg', cv.IMREAD_GRAYSCALE)
    #img = cv.imread('img_data/naturalezaM.jpg', cv.IMREAD_GRAYSCALE)
    #img = cv.imread('img_data/uvas.jpg', cv.IMREAD_GRAYSCALE)

    img = cv.resize(img, (600, 400))

    threshold_otsu = lip.OtsuThreshold(img)
    threshold_ridler_calvard = lip.RidlerCalvardThreshold(img)
    threshold_hysteresis = lip.HysteresisThreshold(img, 120, 115) # optimal image img2.jpeg
    #threshold_hysteresis = lip.HysteresisThreshold(img, 25, 170) # optimal image caballos.webp
    #threshold_hysteresis = lip.HysteresisThreshold(img, 30, 220) # optimal image naturalezaM.jpg
    #threshold_hysteresis = lip.HysteresisThreshold(img, 48, 50) # optimal image uvas.jpg

    # Aplicar umbralización
    _, img_threshold_otsu = cv.threshold(img, threshold_otsu, 255, cv.THRESH_BINARY)
    _, img_threshold_ridler_calvard = cv.threshold(img, threshold_ridler_calvard, 255, cv.THRESH_BINARY)

    lip.show_images_together([img, img_threshold_otsu, img_threshold_ridler_calvard, threshold_hysteresis],
                            ['Original', 'otsu', 'ridler calvard', 'histeresis'])


   
if __name__ == "__main__":
    main()
