import cv2 as cv
import numpy as np
import ena

def linear_contrast_stretch(img):
    # Encontrar los valores mínimo y máximo de la imagen original
    p_min, p_max = np.min(img), np.max(img)

    # Aplicar la fórmula de estiramiento lineal
    stretched = (img - p_min) * (255.0 / (p_max - p_min))
    stretched = np.clip(stretched, 0, 255)  # Limitar valores entre 0 y 255
    return stretched.astype(np.uint8)

def main():
    # Leer la imagen en escala de grises
    path = ena.parse_args_path()

    img = cv.imread(path, cv.IMREAD_GRAYSCALE)

    # Aplicar Linear Contrast Stretch
    img_stretched = linear_contrast_stretch(img)

    # Mostrar las imágenes
    cv.imshow("Original", img)
    cv.imshow("Contrast Stretched", img_stretched)

    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
