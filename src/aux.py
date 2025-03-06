import numpy as np
import cv2

def weighted_guided_filter(I, p, r, eps, W):
    """
    Aplica el filtro Weighted Guided Image Filter (WGIF).

    Parámetros:
    - I: Imagen guía (generalmente la imagen de entrada).
    - p: Imagen de entrada a filtrar.
    - r: Radio del filtro.
    - eps: Parámetro de regularización para evitar divisiones por cero.
    - W: Mapa de pesos (para mejorar preservación de bordes).

    Retorna:
    - Imagen filtrada.
    """
    I = I.astype(np.float32) / 255.0
    p = p.astype(np.float32) / 255.0
    W = W.astype(np.float32)

    kernel_size = 2 * r + 1
    ones = np.ones_like(I)

    # Suma local ponderada
    sum_W = cv2.boxFilter(W, -1, (kernel_size, kernel_size))
    sum_IW = cv2.boxFilter(W * I, -1, (kernel_size, kernel_size))
    sum_pW = cv2.boxFilter(W * p, -1, (kernel_size, kernel_size))
    sum_IpW = cv2.boxFilter(W * I * p, -1, (kernel_size, kernel_size))
    sum_IIW = cv2.boxFilter(W * I * I, -1, (kernel_size, kernel_size))

    # Promedios ponderados
    mean_I = sum_IW / sum_W
    mean_p = sum_pW / sum_W
    mean_Ip = sum_IpW / sum_W
    mean_II = sum_IIW / sum_W

    # Covarianza y varianza local ponderada
    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = mean_II - mean_I * mean_I

    # Cálculo del coeficiente a
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    # Promedio ponderado de a y b
    sum_aW = cv2.boxFilter(W * a, -1, (kernel_size, kernel_size))
    sum_bW = cv2.boxFilter(W * b, -1, (kernel_size, kernel_size))
    mean_a = sum_aW / sum_W
    mean_b = sum_bW / sum_W

    # Imagen filtrada
    q = mean_a * I + mean_b
    return (q * 255).astype(np.uint8)


# Cargar imagen en escala de grises
image = cv2.imread('img_data/salt_papper.png', cv2.IMREAD_GRAYSCALE)

# Crear mapa de pesos basado en el gradiente (bordes)
grad_x = cv2.Sobel(image, cv2.CV_32F, 5, 0, ksize=7)
grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 5, ksize=7)
grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
W = np.exp(-grad_magnitude / (grad_magnitude.max() + 1e-6))

# Aplicar filtro WGIF
filtered_img = weighted_guided_filter(image, image, r=1, eps=0.4, W=W)

# Mostrar resultados
cv2.imshow("Imagen Original", image)
cv2.imshow("Imagen Filtrada (WGIF)", filtered_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
