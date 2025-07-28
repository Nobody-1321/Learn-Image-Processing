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


    # ============================================================
    # === Ideal High-Pass Filter ===
    # ============================================================

    # Create ideal high-pass filter kernel
    ideal_kernel = lip.CreateIdealHighpassFilter(image.shape, cutoff_frequency=70)

    # Apply ideal high-pass filter
    ideal_filtered = lip.ApplyFrequencyDomainFilter(image, ideal_kernel)

    # Compute magnitude and phase spectra
    ideal_mag, ideal_phase = lip.ComputeFourierSpectra(ideal_filtered)
    
    # Display results
    combined_ideal = np.hstack((ideal_filtered, ideal_mag, ideal_phase))
    

    cv.imshow('Filtered Image - Ideal Highpass', combined_ideal)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imwrite("ideal_highpass.jpg", combined_ideal)

    # ============================================================
    # === Gaussian High-Pass Filter ===
    # ============================================================

    # Create Gaussian high-pass filter kernel
    gaussian_kernel = lip.CreateGaussianHighpassFilter(image.shape, cutoff_frequency=70)

    # Apply Gaussian high-pass filter
    gaussian_filtered = lip.ApplyFrequencyDomainFilter(image, gaussian_kernel)

    # Compute magnitude and phase spectra
    gaussian_mag, gaussian_phase = lip.ComputeFourierSpectra(gaussian_filtered)

    # Display results
    combined_gaussian = np.hstack((gaussian_filtered, gaussian_mag, gaussian_phase))

    cv.imshow('Filtered Image - Gaussian Highpass', combined_gaussian)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imwrite("gaussian_highpass.jpg", combined_gaussian)

    # ============================================================
    # === Butterworth High-Pass Filter ===
    # ============================================================

    # Create Butterworth high-pass filter kernel (order = 2)
    butter_kernel = lip.CreateButterworthHighpassFilter(image.shape, cutoff_frequency=70, order=2)

    # Apply Butterworth high-pass filter
    butter_filtered = lip.ApplyFrequencyDomainFilter(image, butter_kernel)

    # Compute magnitude and phase spectra
    butter_mag, butter_phase = lip.ComputeFourierSpectra(butter_filtered)

    # Display results
    combined_butter = np.hstack((butter_filtered, butter_mag, butter_phase))

    cv.imshow('Filtered Image - Butterworth Highpass', combined_butter)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imwrite("butterworth_highpass.jpg", combined_butter)

if __name__ == "__main__":
    main()
