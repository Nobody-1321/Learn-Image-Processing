import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Cargar las imágenes
img2 = cv.imread("img_data/flash_ambient/7_ambient.jpg")
img2 = cv.resize(img2, (0, 0), fx=0.3, fy=0.3)  # Redimensionar para que coincida con el tamaño de la imagen flash
img1 = cv.imread("img_data/flash_ambient/7_flash.jpg")
img1 = cv.resize(img1, (0, 0), fx=0.3, fy=0.3)  # Redimensionar para que coincida con el tamaño de la imagen ambient
img3 = cv.imread("enhanced_image.png")

# Comprobar si las imágenes se cargaron correctamente
if img1 is None or img2 is None or img3 is None:
    print("Error: No se pudieron cargar una o más imágenes.")
    exit()

# Asegurar que todas las imágenes tengan el mismo ancho
max_width = max(img1.shape[1], img2.shape[1], img3.shape[1])

# Redimensionar imágenes si es necesario
def resize_to_width(image, width):
    h, w = image.shape[:2]
    ratio = width / w
    return cv.resize(image, (width, int(h * ratio)))

img1 = resize_to_width(img1, max_width)
img2 = resize_to_width(img2, max_width)
img3 = resize_to_width(img3, max_width)

# Crear imagen vertical (una encima de otra)
vertical_image = np.hstack((img1, img2, img3))

# Mostrar el resultado
cv.imshow('Vertical Stack', vertical_image)
cv.waitKey(0)
cv.destroyAllWindows()

# Guardar resultado
cv.imwrite("hstack_4.jpg", vertical_image)