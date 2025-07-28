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
    image = cv.resize(image, (700, 700))

    log_kernel = lip.CreateLaplacianOfGaussianFilter(image.shape, 50)

    # Apply Laplacian of Gaussian filter
    log_filtered = lip.ApplyFrequencyDomainFilter(image, log_kernel)

    # Compute magnitude and phase spectra
    log_mag, log_phase = lip.ComputeFourierSpectra(log_filtered)

    # Display results
    combined_log = np.hstack((log_filtered, log_mag, log_phase))

    cv.imshow('Filtered Image - Log Highpass', combined_log)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    #cv.imwrite("log_highpass.jpg", combined_log)

if __name__ == "__main__":
    main()
