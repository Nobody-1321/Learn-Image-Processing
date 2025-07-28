import cv2 as cv
import numpy as np
import lip

def main():
    # === Load and Prepare Image ===
    # Load image in grayscale
    image = cv.imread("img_data/lena.jpg", cv.IMREAD_GRAYSCALE)
    
    if image is None:
        raise Exception("Error loading the image")

    # Resize image to 700x700 pixels
    #image = cv.resize(image, (700, 700))

    image_r = lip.HomomorphicFilter(image, gammaL=1.02, gammaH=1.5, sigma=50)

    combined = np.hstack((image, image_r))

    cv.imshow("Homomorphic Filtered Image", combined)
    cv.waitKey(0)
    cv.destroyAllWindows()


    image = cv.imread("img_data/nubes.jpeg")
    image_r = lip.HomomorphicFilterLab(image, gammaL=1.012, gammaH=1.6, sigma=80)
    combined_bgr = np.hstack((image, image_r))

    cv.imshow("Homomorphic Filtered BGR Image", combined_bgr)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imwrite("homomorphic_filtered_bgr.jpg", combined_bgr)

if __name__ == "__main__":
    main()