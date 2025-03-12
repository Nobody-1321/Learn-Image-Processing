import cv2 as cv
import numpy as np
import lip 

# Global variables
constant = 1
value_a = 0
value_b = 0

def update_image(*args):
    """
    Updates the displayed image by applying the transformations based on the 
    current values of the trackbars.

    The transformation follows the equation:
        output_image = (constant * input_image) - value_a + value_b

    The result is clipped to the range [0, 255] to ensure valid pixel values.
    """
    global img, imgCopy, constant, value_a, value_b

    # Get the current values from the trackbars
    constant = cv.getTrackbarPos('Constant', 'ImgCopy')
    value_a = cv.getTrackbarPos('Value A', 'ImgCopy')
    value_b = cv.getTrackbarPos('Value B', 'ImgCopy')

    # Apply the transformation
    img_temp = (constant * img.astype(np.int16)) - value_a + value_b
    img_temp = np.clip(img_temp, 0, 255)  # Ensure pixel values stay within [0, 255]
    imgCopy = img_temp.astype(np.uint8)

    # Display the updated image
    cv.imshow('ImgCopy', imgCopy)

def main():
    """
    Loads an image in grayscale, initializes a copy, and creates an interactive
    window with trackbars to adjust the transformation parameters.
    """
    global img, imgCopy

    # Load the image in grayscale
    img = cv.imread("img_data/white_monta.jpg")
    img = lip.BgrToGray(img)

    if img is None:
        print("Error: Unable to load image. Check the file path.")
        return

    # Create an initial copy of the image
    imgCopy = img.copy()

    # Create a window
    cv.namedWindow('ImgCopy', cv.WINDOW_NORMAL)
    cv.resizeWindow('ImgCopy', 800, 600)

    # Create trackbars
    cv.createTrackbar('Value A', 'ImgCopy', 0, 255, update_image)
    cv.createTrackbar('Value B', 'ImgCopy', 0, 255, update_image)
    cv.createTrackbar('Constant', 'ImgCopy', 1, 5, update_image)  # Small range to prevent extreme values

    # Initial update to show the original image
    update_image()

    # Wait for user interaction
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
