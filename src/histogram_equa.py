import cv2 as cv
import numpy as np
import lip


def main():
    # Leer la imagen en escala de grises
    img = cv.imread("img_data/Moises.jpg", cv.IMREAD_GRAYSCALE)
    #img = cv.imread("img_data/white_monta.jpg", cv.IMREAD_GRAYSCALE)
    #img = cv.imread("img_data/ciervo.jpg")
    #img = cv.imread("img_data/noct.jpeg", cv.IMREAD_GRAYSCALE)
    #img = cv.imread("img_data/Rose.jpg", cv.IMREAD_GRAYSCALE)
    #img = cv.imread("img_data/compressed.jpeg", cv.IMREAD_GRAYSCALE)
    #img = cv.imread("img_data/sky.webp")
    img = cv.resize(img, (600, 600), interpolation=cv.INTER_AREA)
    

    img_eq_lip = lip.HistogramEqualization(img)
    
    #lip.compare_histograms(img, img_eq_lip, "Original", "img_eq_lip")
    #lip.show_images_together([img, img_eq_lip],[ ".", "."])

    lip.plot_two_images_and_histograms(img, img_eq_lip, "Original", "img_eq_lip")

if __name__ == "__main__":
    main()
