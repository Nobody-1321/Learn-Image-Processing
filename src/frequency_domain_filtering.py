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
    # === Ideal Low-Pass Filter ===
    # ============================================================

    # Create ideal low-pass filter kernel
    ideal_kernel = lip.CreateIdealLowpassFilter(image.shape, cutoff_frequency=50)

    # Apply ideal low-pass filter
    ideal_filtered = lip.ApplyFrequencyDomainFilter(image, ideal_kernel)

    # Compute magnitude and phase spectra
    ideal_mag, ideal_phase = lip.ComputeFourierSpectra(ideal_filtered)
    orig_mag, orig_phase = lip.ComputeFourierSpectra(image)

    # Display results
    combined_ideal = np.hstack((ideal_filtered, ideal_mag, ideal_phase))
    combined_original = np.hstack((image, orig_mag, orig_phase))

    cv.imshow('Filtered Image - Ideal Lowpass', combined_ideal)
    cv.imshow('Original Image - Fourier Spectra', combined_original)
    cv.waitKey(0)
    cv.destroyAllWindows()
    

    # ============================================================
    # === Gaussian Low-Pass Filter ===
    # ============================================================

    # Create Gaussian low-pass filter kernel
    gaussian_kernel = lip.CreateGaussianLowpassFilter(image.shape, cutoff_frequency=50)

    # Apply Gaussian low-pass filter
    gaussian_filtered = lip.ApplyFrequencyDomainFilter(image, gaussian_kernel)

    # Compute magnitude and phase spectra
    gaussian_mag, gaussian_phase = lip.ComputeFourierSpectra(gaussian_filtered)
    orig_mag, orig_phase = lip.ComputeFourierSpectra(image)

    # Display results
    combined_gaussian = np.hstack((gaussian_filtered, gaussian_mag, gaussian_phase))

    cv.imshow('Filtered Image - Gaussian Lowpass', combined_gaussian)
    cv.imshow('Original Image - Fourier Spectra', combined_original)
    cv.waitKey(0)
    cv.destroyAllWindows()


    # ============================================================
    # === Butterworth Low-Pass Filter ===
    # ============================================================

    # Create Butterworth low-pass filter kernel (order = 2)
    butter_kernel = lip.CreateButterworthLowpassFilter(image.shape, cutoff_frequency=50, order=2)

    # Apply Butterworth low-pass filter
    butter_filtered = lip.ApplyFrequencyDomainFilter(image, butter_kernel)

    # Compute magnitude and phase spectra
    butter_mag, butter_phase = lip.ComputeFourierSpectra(butter_filtered)
    orig_mag, orig_phase = lip.ComputeFourierSpectra(image)

    # Display results
    combined_butter = np.hstack((butter_filtered, butter_mag, butter_phase))

    cv.imshow('Filtered Image - Butterworth Lowpass', combined_butter)
    cv.imshow('Original Image - Fourier Spectra', combined_original)
    cv.waitKey(0)
    cv.destroyAllWindows()


    # ============================================================
    # === Lanczos Low-Pass Filter ===
    # ============================================================

    # Create Lanczos low-pass filter kernel (window parameter a = 2)
    lanczos_kernel = lip.CreateLanczosLowpassFilter(image.shape, cutoff_frequency=50, a=2)

    # Apply Lanczos low-pass filter
    lanczos_filtered = lip.ApplyFrequencyDomainFilter(image, lanczos_kernel)

    # Compute magnitude and phase spectra
    lanczos_mag, lanczos_phase = lip.ComputeFourierSpectra(lanczos_filtered)
    orig_mag, orig_phase = lip.ComputeFourierSpectra(image)

    # Display results
    combined_lanczos = np.hstack((lanczos_filtered, lanczos_mag, lanczos_phase))

    cv.imshow('Filtered Image - Lanczos Lowpass', combined_lanczos)
    cv.imshow('Original Image - Fourier Spectra', combined_original)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
