import numpy as np
import cv2 as cv

# Matriz de ejemplo con valores arbitrarios
Gx = np.array([
    [-100, 50, 200],
    [30, 400, -50],
    [0, 255, 500]
], dtype=np.float32)

# Normalización con cv.NORM_MINMAX (escala a [0, 255])
Gx_norm = cv.normalize(Gx, None, 0, 255, cv.NORM_MINMAX)

print("Matriz original:\n", Gx)
print("\nMatriz normalizada:\n", Gx_norm.astype(np.uint8))  # Convertimos a uint8 para mostrar mejor
