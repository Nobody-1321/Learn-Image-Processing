import cv2 as cv
import numpy as np
import lip  

def main():
    # === Load and Prepare Image ===
    # Load image in grayscale

    image = cv.imread("img_data/lienzo.jpg", cv.IMREAD_GRAYSCALE)
    
    if image is None:
        raise Exception("Error loading the image")

    # Resize image to 700x700 pixels
    #image = cv.resize(image, (700, 700))

    um_kernel = lip.CreateUnsharpMaskingFilter(image.shape, 100, alpha=1.0, method='butterworth')

    um_filtered = lip.ApplyFrequencyDomainFilter(image, um_kernel)

    um_mag, um_phase = lip.ComputeFourierSpectra(um_filtered)

    combined_um = np.hstack((um_filtered, um_mag, um_phase))
    combined_images = np.hstack((image, um_filtered))
    
    cv.imshow('Original and Filtered Image - Unsharp Masking', combined_images)
    cv.imshow('Filtered Image - Unsharp Masking', combined_um)
    cv.waitKey(0)
    cv.destroyAllWindows()

    image = cv.imread("img_data/roma.jpeg")
    height, width = image.shape[:2]

    um_kernel_bgr = lip.CreateUnsharpMaskingFilter((height, width), 90, alpha=1.0, method='butterworth')
    um_filtered_bgr = lip.ApplyFrequencyDomainFilterLabL(image, um_kernel_bgr)
    
    combined_images_bgr = np.hstack((image, um_filtered_bgr))
    cv.imshow('Original and Filtered Image - Unsharp Masking BGR', combined_images_bgr)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imwrite("log_highpass.jpg", combined_images_bgr)

if __name__ == "__main__":
    main()
