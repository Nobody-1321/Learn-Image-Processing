import ena
import numpy as np
import cv2 as cv

def main():

    #img = cv.imread("img_data/caballo.webp")
    img = cv.imread("img_data/paper.png")
    #img = cv.imread("img_data/astro.jpg")
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, (400, 400))

    #img_noise = ena.MedianFilter(img, 3)
    #img_noise = ena.NonLocalMeans(img, 3, 3, 0.33)
    ss = 1 # Suavizado espacial
    sr = 0.333  # Suavizado en intensidad
    niter = 2 # Número de iteraciones
    img_noise = ena.BilateralFilterFast(img, ss, sr, niter)
    

    

    ena.show_two_images_side_by_side(img, img_noise, "Original vs Noise", "horizontal")
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()