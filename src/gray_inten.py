import cv2 as cv
import numpy as np
import ena

# Variables globales
constante = 1
valor_a = 0
valor_b = 0

def update_image(*args):
    """
    Actualiza la imagen combinando los valores de todas las barras deslizantes.
    """
    global img, imgCopy, constante, valor_a, valor_b

    # Obtener los valores actuales de las barras
    constante = cv.getTrackbarPos('Constante', 'ImgCopy')
    valor_a = cv.getTrackbarPos('Valor A', 'ImgCopy')
    valor_b = cv.getTrackbarPos('Valor B', 'ImgCopy')

    # Operaciones combinadas
    img_temp = (constante * img.astype(np.int16)) - valor_a + valor_b
    img_temp = np.clip(img_temp, 0, 255)  # Limitar entre 0 y 255
    imgCopy = img_temp.astype(np.uint8)

    # Mostrar la imagen actualizada
    cv.imshow('ImgCopy', imgCopy)

def main():
    global img, imgCopy

    # Cargar la imagen
    img = cv.imread("img_data/astro.jpg",cv.IMREAD_GRAYSCALE)

    # Copia inicial de la imagen
    imgCopy = img.copy()

    # Crear ventana
    cv.namedWindow('ImgCopy', cv.WINDOW_NORMAL)
    cv.resizeWindow('ImgCopy', 800, 600)

    # Crear barras deslizantes
    cv.createTrackbar('Valor A', 'ImgCopy', 0, 255, update_image)
    cv.createTrackbar('Valor B', 'ImgCopy', 0, 255, update_image)
    cv.createTrackbar('Constante', 'ImgCopy', 1, 5, update_image)  # Rango pequeño para evitar valores grandes

    # Actualización inicial
    update_image()

    # Esperar a que el usuario cierre la ventana
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
