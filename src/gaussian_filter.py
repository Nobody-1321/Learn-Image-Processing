# This script demonstrates the use of Gaussian smoothing and edge detection on images.
# It applies Gaussian smoothing to a halftone image and a Lena image, then performs
# Canny edge detection on the smoothed and original images.
# It also demonstrates how to extract an object from an image using a binary mask and combine it with a blurred background.

import cv2
import matplotlib.pyplot as plt
import lip

def main():
    
    # Load the first image (halftone style)
    img = cv2.imread("img_data/halftone_2.jpeg", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (720, 720))

    # Apply Gaussian smoothing
    # NOTE: The 'GaussianFilterGrayscale' function is slow because it is not optimized for performance.
    img_smoothed = lip.GaussianFilterGrayscale(img, 2.5)

    # Display original and smoothed images along with their histograms
    lip.plot_two_images_and_histograms(img, img_smoothed, "Original", "Smoothed")
    
    # Load the second image (Lena image)
    img2 = cv2.imread("img_data/lena.jpg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.resize(img2, (720, 720))
    
    # Apply Gaussian smoothing again (slow operation)
    img2_smoothed = lip.GaussianFilterGrayscale(img2, 2.5)

    # Apply Canny edge detection on the smoothed image
    edges = cv2.Canny(img2_smoothed, threshold1=80, threshold2=90)

    # Apply Canny edge detection directly on the original (unsmoothed) image
    edges_without_smoothed = cv2.Canny(img2, threshold1=40, threshold2=120)
    
    # Display the edge detection results side by side
    lip.show_images_together([edges, edges_without_smoothed], [".", "."])
    
    # Load the third image (Lena in color)
    img3 = cv2.imread("img_data/lena.jpg")
    img3 = cv2.resize(img3, (720, 720))

    # Apply strong Gaussian smoothing to the background
    # NOTE: The 'GaussianFilterRGB' function is slow because it is not optimized for performance.
    img3_smoothed = lip.GaussianFilterBGR(img3, 15.0)
    
    # Load a binary mask (white for the object, black for the background)
    # Important: Load the mask in grayscale mode
    mask = cv2.imread("img_data/masklena.png", cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (720, 720))

    # Extract the object (Lena) from the original image using the mask
    result_lena = cv2.bitwise_and(img3, img3, mask=mask)

    # Invert the mask to select the background
    inverted_mask = cv2.bitwise_not(mask)

    # Extract the heavily smoothed background
    result_background = cv2.bitwise_and(img3_smoothed, img3_smoothed, mask=inverted_mask)

    # Combine the sharp object and the blurred background
    result = cv2.add(result_lena, result_background)
    # save the result
    cv2.imwrite("img_data/lena_composite.png", result)
    # Display the final composited result
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
if __name__ == "__main__":
    main()
