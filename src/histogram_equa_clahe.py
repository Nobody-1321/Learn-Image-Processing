import numpy as np
import cv2 
import lip

# Ejemplo de uso:
if __name__ == "__main__":
    
    # Cargar la imagen en escala de grises (asegúrate de tener una imagen 'imagen.jpg' en tu directorio)
    #img = cv.imread("img_data/Moises.jpg", cv.IMREAD_GRAYSCALE)
    #img = cv2.imread("img_data/white_monta.jpg", cv2.IMREAD_GRAYSCALE)
    #img = cv2.imread("img_data/ciervo.jpg", cv2.IMREAD_GRAYSCALE)
    img = cv2.imread("img_data/noct.jpeg", cv2.IMREAD_GRAYSCALE)
    #img = cv.imread("img_data/Rose.jpg", cv.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (600, 600), interpolation=cv2.INTER_AREA)
    
    if img is None:
        raise ValueError("No se pudo cargar la imagen. Verifica la ruta.")
    
    # Aplicar CLAHE
    img_equ = lip.HistogramEqualization(img)
    img_clahe = lip.HistogramEqualizationClahe(img, clip_limit=75, grid_size=(13, 13))
    