import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Cargar imagen en escala de grises
#img = cv2.imread('img_data/acat.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('img_data/desert1.jpg', cv2.IMREAD_GRAYSCALE)

# 2. Calcular la DFT 2D
dft = np.fft.fft2(img)

# 3. Separar magnitud y fase
magnitude = np.abs(dft)
phase = np.angle(dft)

# 4. Aplicar logaritmo a la magnitud para mejorar visualización
log_magnitude = np.log1p(magnitude)

# 5. Centrar la DFT (llevar el componente DC al centro)
log_magnitude_shifted = np.fft.fftshift(log_magnitude)
phase_shifted = np.fft.fftshift(phase)

# 6. Escalar fase al rango [0, 255] para visualizarla como imagen
phase_normalized = cv2.normalize(phase_shifted, None, 0, 255, cv2.NORM_MINMAX)
phase_normalized = phase_normalized.astype(np.uint8)

# 7. Mostrar resultados
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title('Imagen original')
plt.imshow(img, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Log Magnitud (centrada)')
plt.imshow(log_magnitude_shifted, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Fase (normalizada)')
plt.imshow(phase_normalized, cmap='gray')

plt.tight_layout()
plt.show()
