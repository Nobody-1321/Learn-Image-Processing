import cv2
import numpy as np
import matplotlib.pyplot as plt
import lip


# Cargar imagen en escala de grises
#img = cv2.imread("img_data/caballo.webp", cv2.IMREAD_GRAYSCALE)
#img = cv2.imread("img_data/Moises.jpg", cv2.IMREAD_GRAYSCALE)
#img = cv2.imread("img_data/white_monta.jpg", cv2.IMREAD_GRAYSCALE)
#img = cv2.imread("img_data/noct.jpeg", cv2.IMREAD_GRAYSCALE)
#img = cv2.imread("img_data/Rose.jpg", cv2.IMREAD_GRAYSCALE)
#img = cv2.imread("img_data/compressed.jpeg", cv2.IMREAD_GRAYSCALE)
#img = cv2.imread("img_data/sky.webp", cv2.IMREAD_GRAYSCALE)
#img = cv2.imread("img_data/caballo.webp", cv2.IMREAD_GRAYSCALE)
#img = cv2.imread("img_data/ciervo.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.imread("img_data/acat.jpg", cv2.IMREAD_GRAYSCALE)

# Validar si la imagen se cargó correctamente
if img is None:
    raise ValueError("No se pudo cargar la imagen. Verifica la ruta del archivo.")

# Aplicar Equalización del Bihistograma
bhe_grayscale = lip.BihistogramEqualizationGrayscale(img)

# Mostrar resultados
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Imagen Original")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Equalización del Bihistograma")
plt.imshow(bhe_grayscale, cmap="gray")
plt.axis("off")

plt.show()