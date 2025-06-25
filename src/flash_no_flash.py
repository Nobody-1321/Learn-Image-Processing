import cv2
import numpy as np

def bilateral_filter(image, sigma_d=4, sigma_r=0.03):
    return cv2.bilateralFilter(image, d=-1, sigmaColor=sigma_r*255, sigmaSpace=sigma_d)

def joint_bilateral_filter(ambient, flash, sigma_d=6, sigma_r=0.1):
    import cv2.ximgproc  # Asegúrate de tener opencv-contrib-python

    ambient_uint8 = ambient.astype(np.uint8)
    flash_uint8 = flash.astype(np.uint8)

    filtered = np.zeros_like(ambient_uint8)
    
    for i in range(3):  # Procesamos por canal (BGR)
        filtered[..., i] = cv2.ximgproc.jointBilateralFilter(
            joint=flash_uint8[..., i],    # Imagen guía (con flash)
            src=ambient_uint8[..., i],    # Imagen que se filtra (sin flash)
            d=-1,
            sigmaColor=sigma_r * 255,
            sigmaSpace=sigma_d
        )

    return filtered.astype(np.float32)

def compute_detail_layer(flash, sigma_d=30, sigma_r=0.9, epsilon=0.5):
    base = bilateral_filter(flash, sigma_d, sigma_r)
    detail = (flash + epsilon) / (base + epsilon)
    return detail

def apply_masked_merge(a, b, mask):
    return (1 - mask) * a + mask * b

def detect_flash_shadows(flash_lin, ambient_lin, tau=0.05):
    diff = flash_lin - ambient_lin
    shadow_mask = np.all(diff < tau, axis=2).astype(np.float32)
    return cv2.dilate(shadow_mask, None)

def enhance_ambient_with_flash(ambient, flash):
    # Suavizado y denoising
    base_ambient = bilateral_filter(ambient, sigma_d=16, sigma_r=0.05)
    denoised_ambient = joint_bilateral_filter(ambient, flash)

    # Cálculo de capa de detalle
    detail_layer = compute_detail_layer(flash)

    # Detección de sombras del flash
    ambient_lin = ambient.astype(np.float32) / 255
    flash_lin = flash.astype(np.float32) / 255
    mask = detect_flash_shadows(flash_lin, ambient_lin)
    mask = cv2.GaussianBlur(mask, (11, 11), 5)
    mask = np.repeat(mask[..., np.newaxis], 3, axis=2)  # expand to 3 channels

    # Fusión final
    transferred = denoised_ambient * detail_layer
    final_image = apply_masked_merge(transferred, base_ambient, mask)

    return np.clip(final_image, 0, 255).astype(np.uint8)

# Cargar imágenes
ambient = cv2.imread('img_data/flash_ambient/3_ambient.jpeg').astype(np.float32)
flash = cv2.imread('img_data/flash_ambient/3_flash.jpeg').astype(np.float32)

# Redimensionar si no están alineadas
ambient = cv2.resize(ambient, (flash.shape[1], flash.shape[0]))

# Aplicar mejora
enhanced = enhance_ambient_with_flash(ambient, flash)
cv2.imwrite('enhanced_image.png', enhanced)
print("Imagen mejorada guardada como 'enhanced_image.png'.")

