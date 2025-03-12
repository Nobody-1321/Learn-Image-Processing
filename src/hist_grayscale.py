import cv2 as cv
import numpy as np
import lip

def main():
    dark_img = cv.imread("img_data/noct.jpeg", cv.IMREAD_GRAYSCALE)
    dark_img = cv.resize(dark_img, (800, 600))
    bright_img = cv.imread("img_data/ciervo1.png", cv.IMREAD_GRAYSCALE)
    bright_img = cv.resize(bright_img, (800, 600))
    low_contrast_img = cv.imread("img_data/white_monta1.png", cv.IMREAD_GRAYSCALE)
    low_contrast_img = cv.resize(low_contrast_img, (800, 600))

    if dark_img is None or bright_img is None:
        exit("Error: Unable to load image. Check the file path.")   

    lip.plot_three_images_and_histograms(dark_img,  low_contrast_img, bright_img, "Image 1", "Image 2", "Image 3")

if __name__ == "__main__":
    main()

