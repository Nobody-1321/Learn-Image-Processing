import cv2 as cv
import numpy as np
import lip

def main():
    # === Load and Prepare Image ===
    # Load image in BGR
    image = cv.imread("img_data/caballos.jpeg")
    if image is None:
        raise Exception("Error loading the image")

    # Resize image to 700x700 pixels
    image = cv.resize(image, (700, 680))

    # Extract height and width
    height, width = image.shape[:2]

    # ============================================================
    # === Ideal High-Pass Filter ===
    # ============================================================

    ideal_kernel = lip.CreateIdealLowpassFilter((height, width), cutoff_frequency=50)
    ideal_filtered = lip.ApplyFrequencyDomainFilterBGR(image, ideal_kernel)
    
    # ============================================================
    # === Gaussian High-Pass Filter ===
    # ============================================================

    gaussian_kernel = lip.CreateGaussianLowpassFilter((height, width), cutoff_frequency=50)
    gaussian_filtered = lip.ApplyFrequencyDomainFilterBGR(image, gaussian_kernel)

    # ============================================================
    # === Butterworth High-Pass Filter ===
    # ============================================================

    butter_kernel = lip.CreateButterworthLowpassFilter((height, width), cutoff_frequency=50, order=1)
    butter_filtered = lip.ApplyFrequencyDomainFilterBGR(image, butter_kernel)

    # ============================================================
    # === Display Results ===
    # ============================================================

    combined = np.hstack((ideal_filtered, gaussian_filtered, butter_filtered))
    cv.imshow("High-Pass Filter Results", combined)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":  
    main()