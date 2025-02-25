import cv2 as cv
import numpy as np

def apply_dilation(image, kernel_size=(5, 5)):
    kernel = np.ones(kernel_size, np.uint8)
    dilated_image = cv.dilate(image, kernel, iterations=10)
    return dilated_image

def apply_erosion(image, kernel_size=(5, 5)):
    kernel = np.ones(kernel_size, np.uint8)
    eroded_image = cv.erode(image, kernel, iterations=10)
    return eroded_image

# Example usage
if __name__ == "__main__":
    # Load the image
    img = cv.imread('img_data/drawn_image_2.png', cv.IMREAD_GRAYSCALE)

    # Apply dilation
    dilated_img = apply_dilation(img)
    cv.imshow('Dilated Image', dilated_img)

    # Apply erosion
    eroded_img = apply_erosion(img)
    cv.imshow('Eroded Image', eroded_img)

    # Display the original image
    cv.imshow('Original Image', img)

    # Wait for a key press and close all windows
    cv.waitKey(0)
    cv.destroyAllWindows()