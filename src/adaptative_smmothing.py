import numpy as np
import cv2

def compute_gradient_magnitude(I):
    """
    Calcula la magnitud del gradiente de la imagen usando derivadas de Sobel.
    
    Parámetros:
    - I: Imagen de entrada en escala de grises.

    Retorna:
    - Magnitud del gradiente.
    """
    Gx = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=3)  # Gradiente en X
    Gy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=3)  # Gradiente en Y
    return np.sqrt(Gx**2 + Gy**2)  # Magnitud del gradiente

def adaptive_smoothing(I, sigma, window_size=5):
    """
    Aplica un suavizado adaptativo basado en la magnitud del gradiente.

    Parámetros:
    - I: Imagen de entrada en escala de grises (numpy array).
    - sigma: Parámetro de escala para controlar la suavización.
    - window_size: Tamaño de la ventana local (debe ser impar).

    Retorna:
    - Imagen suavizada.
    """
    # Convertir imagen a flotante para evitar problemas de precisión
    I = I.astype(np.float64)
    
    # Calcular magnitud del gradiente
    grad_mag = compute_gradient_magnitude(I)

    # Obtener dimensiones de la imagen
    h, w = I.shape

    # Crear imagen de salida
    I_smooth = np.zeros_like(I)

    # Mitad del tamaño de la ventana
    w_half = window_size // 2

    # Iterar sobre cada píxel
    for y in range(w_half, h - w_half):
        for x in range(w_half, w - w_half):
            # Extraer ventana local
            region = I[y - w_half:y + w_half + 1, x - w_half:x + w_half + 1]
            grad_region = grad_mag[y - w_half:y + w_half + 1, x - w_half:x + w_half + 1]

            # Calcular pesos inversamente proporcionales al gradiente
            weights = np.exp(-grad_region**2 / (2 * sigma**2))

            # Aplicar suavizado adaptativo (fórmula 5.155)
            I_smooth[y, x] = np.sum(region * weights) / np.sum(weights)

    return np.clip(I_smooth, 0, 255).astype(np.uint8)  # Convertir a uint8

# Cargar imagen de prueba
I = cv2.imread("img_data/paper.png", cv2.IMREAD_GRAYSCALE)


# Aplicar suavizado adaptativo
I_smooth = adaptive_smoothing(I, sigma=9)

# Mostrar imágenes
cv2.imshow("Original", I)
cv2.imshow("Adaptive Smoothing", I_smooth)
cv2.waitKey(0)
cv2.destroyAllWindows()
