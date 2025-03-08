import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

def rgb_to_hsi(image):
    """Convierte una imagen RGB a HSI."""
    image = image.astype(np.float32) / 255.0
    R, G, B = cv2.split(image)
    
    # Intensidad
    I = (R + G + B) / 3.0
    
    # Saturación
    min_RGB = np.minimum(np.minimum(R, G), B)
    S = 1 - (3 * min_RGB / (R + G + B + 1e-6))  # Evitar división por cero

    # Tono (Hue)
    numerator = 0.5 * ((R - G) + (R - B))
    denominator = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-6  # Evitar división por cero
    theta = np.arccos(numerator / denominator)  # Ángulo en radianes
    
    H = np.where(B > G, 2 * np.pi - theta, theta)  # Ajustar hue si B > G
    H = H / (2 * np.pi)  # Normalizar a [0, 1]

    return cv2.merge([H, S, I])

def hsi_to_rgb(hsi_image):
    """Convierte una imagen HSI de vuelta a RGB."""
    H, S, I = cv2.split(hsi_image)
    H = H * 2 * np.pi  # Convertir de [0, 1] a [0, 2π]
    
    R, G, B = np.zeros_like(H), np.zeros_like(H), np.zeros_like(H)
    
    # Sector 0 (0° - 120°)
    idx = (H < 2*np.pi/3)
    B[idx] = I[idx] * (1 - S[idx])
    R[idx] = I[idx] * (1 + S[idx] * np.cos(H[idx]) / np.cos(np.pi/3 - H[idx]))
    G[idx] = 3 * I[idx] - (R[idx] + B[idx])

    # Sector 1 (120° - 240°)
    idx = (H >= 2*np.pi/3) & (H < 4*np.pi/3)
    R[idx] = I[idx] * (1 - S[idx])
    G[idx] = I[idx] * (1 + S[idx] * np.cos(H[idx] - 2*np.pi/3) / np.cos(np.pi - H[idx]))
    B[idx] = 3 * I[idx] - (R[idx] + G[idx])

    # Sector 2 (240° - 360°)
    idx = (H >= 4*np.pi/3)
    G[idx] = I[idx] * (1 - S[idx])
    B[idx] = I[idx] * (1 + S[idx] * np.cos(H[idx] - 4*np.pi/3) / np.cos(5*np.pi/3 - H[idx]))
    R[idx] = 3 * I[idx] - (G[idx] + B[idx])

    rgb_image = cv2.merge([R, G, B])
    return np.clip(rgb_image * 255.0, 0, 255).astype(np.uint8)

def weighted_guided_filter(I, p, r=5, eps=0.1):
    """Filtro guiado ponderado (WGIF) basado en el artículo."""
    mean_I = gaussian_filter(I, r)
    mean_p = gaussian_filter(p, r)
    corr_I = gaussian_filter(I * I, r)
    corr_Ip = gaussian_filter(I * p, r)

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = gaussian_filter(a, r)
    mean_b = gaussian_filter(b, r)

    q = mean_a * I + mean_b
    return q

def enhance_image(image):
    """Aplica el algoritmo de mejora de imagen basado en WGIF y Retinex."""
    hsi = rgb_to_hsi(image)
    H, S, I = cv2.split(hsi)

    # Estimación de la iluminación con WGIF
    I_smooth = weighted_guided_filter(I, I, r=5, eps=0.04)

    # Corrección de brillo con Gamma Adaptativo
    a = np.mean(I_smooth)
    gamma_corrected = (I_smooth + a) / (1.3 + a)
    
    # Suavizado del componente de reflexión
    R = I / (I_smooth + 1e-6)  # Evitar división por cero
    R_smooth = weighted_guided_filter(R, R, r=5, eps=0.036)

    # Fusión de la imagen corregida
    I_enhanced = gamma_corrected * R_smooth

    # Restauración del color
    alpha = I_enhanced / (I + 1e-6)

    R_new = np.clip(alpha * image[:, :, 0], 0, 255).astype(np.uint8)
    G_new = np.clip(alpha * image[:, :, 1], 0, 255).astype(np.uint8)
    B_new = np.clip(alpha * image[:, :, 2], 0, 255).astype(np.uint8)

    enhanced_image = cv2.merge([R_new, G_new, B_new])
    return enhanced_image

# Prueba con una imagen de ejemplo
#image = cv2.imread("img_data/lowlight.jpg")  # Cambiar por la imagen real
image = cv2.imread("img_data/blue_dark.jpg")  # Cambiar por la imagen real

image = cv2.resize(image, (640, 480))  # Redimensionar para visualización
if image is not None:
    enhanced_image = enhance_image(image)
    cv2.imwrite("enhanced_example1.jpg", enhanced_image)

"Implementación completada."
