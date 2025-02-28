
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import ena


def main():
    # Leer la imagen en escala de grises
    img = cv.imread('Y4.jpg', cv.COLOR_BGR2GRAY)
    
    # Aplicar ecualización de histograma manual
    img_equalized_manual = ena.histogram_equalization(img)
    
    # Aplicar ecualización usando OpenCV
    img_equalized_cv = cv.equalizeHist(img)



    # Mostrar con OpenCV para comparar resultados
    cv.imshow('Imagen Original', img)
    cv.imshow('Ecualizacin Manual', img_equalized_manual)
    cv.imshow('Ecualizacin OpenCV', img_equalized_cv)
    
    cv.waitKey(0)
    cv.destroyAllWindows()


    exit()
    # Visualizar resultados
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.title("Imagen Original")
    plt.imshow(img, cmap='gray')
    plt.axis("off")
    
    plt.subplot(2, 2, 2)
    plt.title("Histograma Original")
    plt.hist(img.flatten(), bins=256, range=[0, 256], color='black')
    
    plt.subplot(2, 2, 3)
    plt.title("Ecualización Manual")
    plt.imshow(img_equalized_manual, cmap='gray')
    plt.axis("off")
    
    plt.subplot(2, 2, 4)
    plt.title("Histograma Ecualizado")
    plt.hist(img_equalized_manual.flatten(), bins=256, range=[0, 256], color='black')
    
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    main()
