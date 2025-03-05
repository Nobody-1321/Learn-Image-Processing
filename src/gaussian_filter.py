import cv2
import matplotlib.pyplot as plt
import ena

def main():

    img = cv2.imread("img_data/salt_papper.png", cv2.IMREAD_GRAYSCALE)
    
    img = cv2.resize(img, (400, 400))

    ena.show_image_with_histogram(img, "Paper Image")

    # Aplicar filtro gaussiano
    img_smoothed = ena.GaussianFilterGrayscale(img, 1.5)
    ena.show_image_with_histogram(img_smoothed, "Smoothed Image")

    # comparar histogramas
    ena.compare_histograms(img, img_smoothed, "Original", "Smoothed")

if __name__ == "__main__":
    main()