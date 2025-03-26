import numpy as np
import cv2 as cv
import lip

# Ejemplo de uso:
if __name__ == "__main__":
    
    # Cargar la imagen en escala de grises (asegúrate de tener una imagen 'imagen.jpg' en tu directorio)
    #img = cv.imread("img_data/Moises.jpg", cv.IMREAD_GRAYSCALE)
    #img = cv.imread("img_data/white_monta.jpg", cv.IMREAD_GRAYSCALE)
    #img = cv.imread("img_data/escena.jpg", cv.IMREAD_GRAYSCALE)
    img = cv.imread("img_data/acat.jpg", cv.IMREAD_GRAYSCALE)
    #img = cv.imread("img_data/ciervo.jpg", cv.IMREAD_GRAYSCALE)
    #img = cv.imread("img_data/noct.jpeg", cv.IMREAD_GRAYSCALE)
    #img = cv.imread("img_data/Rose.jpg", cv.IMREAD_GRAYSCALE)
    
    img = cv.resize(img, (600, 600), interpolation=cv.INTER_AREA)
    
    if img is None:
        raise ValueError("No se pudo cargar la imagen. Verifica la ruta.")
    
    # Aplicar CLAHE
    img_equ = lip.HistogramEqualization(img)
    img_clahe = lip.HistogramEqualizationClaheGrayscale(img, clip_limit=0, grid_size=(9, 9))

    # Mostrar las imágenes
    lip.plot_three_images_and_histograms(img, img_equ, img_clahe, "Original", "Histogram Equalization", "CLAHE")
    
    