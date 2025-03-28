import cv2
import lip
import numpy as np
import matplotlib.pyplot as plt


#img = cv2.imread("img_data/Moises.jpg")
#img = cv2.imread("img_data/white_monta.jpg")
#img = cv2.imread("img_data/noct.jpeg")
#img = cv2.imread("img_data/Rose.jpg")
#img = cv2.imread("img_data/compressed.jpeg")
#img = cv2.imread("img_data/sky.webp")
img = cv2.imread("img_data/caballo.webp")
#img = cv2.imread("img_data/ciervo.jpg")
#img = cv2.imread("img_data/acat.jpg")

# Aplicar Equalización del Bihistograma
bhe_color = lip.BihistogramEqualization(img)

# Mostrar resultados
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Imagen Original")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convertir BGR a RGB para matplotlib
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Equalización del Bihistograma")
plt.imshow(cv2.cvtColor(bhe_color, cv2.COLOR_BGR2RGB))  # Convertir BGR a RGB para matplotlib
plt.axis("off")

plt.show()