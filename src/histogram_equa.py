
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import lip


def main():
    # Leer la imagen en escala de grises
    img = cv.imread('img/Y4.jpg', cv.COLOR_BGR2GRAY)
    
    # Aplicar ecualización de histograma manual
    
    img_histogram = cv.calcHist([img], [0], None, [256], [0, 256])
    img_equalized_manual = lip.HistogramEqualization(img)
    equa_histogram = cv.calcHist([img_equalized_manual], [0], None, [256], [0, 256])
    
    lip.plot_images_and_histograms(img, img_equalized_manual, img_histogram, equa_histogram)
    

if __name__ == "__main__":
    main()
