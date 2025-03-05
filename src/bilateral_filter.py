import cv2
import matplotlib.pyplot as plt
import ena

def main():

    img = cv2.imread("img_data/salt_papper.png", cv2.IMREAD_GRAYSCALE)    
    img = cv2.resize(img, (400, 400))

    # Aplicar filtro gaussiano
    img_smoothed = ena.BilateralFilterFast(img, 1.5, 1.5, 4)
    
    #comparar histogramas
    ena.compare_histograms(img, img_smoothed, "Original", "Smoothed")

    # Mostrar imágenes
    cv2.imshow("Original", img)
    cv2.imshow("Smoothed", img_smoothed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()