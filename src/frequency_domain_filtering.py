import numpy as np
import cv2
import matplotlib.pyplot as plt

# Leer imagen en escala de grises

img = cv2.imread('img_data/lena_salt.jpg', cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape
crow, ccol = rows // 2 , cols // 2  # centro

# FFT
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# Crear una máscara circular ideal lowpass
radius = 30  # radio de corte
mask = np.zeros((rows, cols), np.uint8)
cv2.circle(mask, (ccol, crow), radius, 1, thickness=-1)

# Aplicar filtro en frecuencia
fshift_filtered = fshift * mask

# Transformada inversa
f_ishift = np.fft.ifftshift(fshift_filtered)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

magnitude_spectrum = 20 * np.log(np.abs(fshift))
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitud del espectro (FFT shift)')

# Mostrar resultados
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.subplot(1, 3, 2), plt.imshow(mask * 255, cmap='gray'), plt.title('Ideal LPF Mask')
plt.subplot(1, 3, 3), plt.imshow(img_back, cmap='gray'), plt.title('Filtrada (Ringing visible)')
plt.show()
