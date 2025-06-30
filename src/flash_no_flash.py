import cv2
import numpy as np

def bilateral_filter(image, sigma_d=1, sigma_r=0.1):
    return cv2.bilateralFilter(image, d=-1, sigmaColor=sigma_r*255, sigmaSpace=sigma_d)

def joint_bilateral_filter(ambient, flash, sigma_d=15, sigma_r=0.1):
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

def compute_detail_layer(flash, sigma_d=10, sigma_r=0.7, epsilon=0.5):
    base = bilateral_filter(flash, sigma_d, sigma_r)
    detail = (flash + epsilon) / (base + epsilon)
    return detail

def apply_masked_merge(a, b, mask):
    return (1 - mask) * a + mask * b

def detect_flash_shadows(flash_lin, ambient_lin, tau=0.09):
    diff = flash_lin - ambient_lin
    shadow_mask = np.all(diff < tau, axis=2).astype(np.float32)
    return cv2.dilate(shadow_mask, None)

def detect_flash_specularities(flash_lin, threshold=0.95):
    # Convertir a luminancia (usamos coeficientes de percepción)
    luminance = 0.2126 * flash_lin[..., 2] + 0.7152 * flash_lin[..., 1] + 0.0722 * flash_lin[..., 0]
    specular_mask = (luminance >= threshold).astype(np.float32)
    return cv2.dilate(specular_mask, None)

def estimate_ambient_color(flash_lin, ambient_lin, tau1=0.02, tau2=0.02):
    delta = flash_lin - ambient_lin
    luminance_delta = 0.2126 * delta[..., 2] + 0.7152 * delta[..., 1] + 0.0722 * delta[..., 0]
    luminance_ambient = 0.2126 * ambient_lin[..., 2] + 0.7152 * ambient_lin[..., 1] + 0.0722 * ambient_lin[..., 0]

    valid = (luminance_delta > tau2) & (luminance_ambient > tau1)
    valid = valid[..., np.newaxis]

    delta_safe = np.where(delta < 1e-4, 1e-4, delta)  # para evitar división por cero
    C = ambient_lin / delta_safe
    C = C * valid

    count = np.maximum(np.sum(valid), 1)
    ambient_color = np.sum(C, axis=(0, 1)) / count
    return ambient_color

def white_balance_by_ambient_color(ambient_lin, ambient_color):
    wb_image = ambient_lin / (ambient_color[None, None, :] + 1e-6)
    return np.clip(wb_image, 0, 1)

def normalize_color(color):
    return color / (np.mean(color) + 1e-6)

def enhance_ambient_with_flash(ambient, flash):

    image_bilateral = bilateral_filter(ambient, sigma_d=15, sigma_r=0.1)
    cv2.imshow('Bilateral Filtered Ambient', image_bilateral.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ambient_lin = ambient.astype(np.float32) / 255
    flash_lin = flash.astype(np.float32) / 255
    
    # --- White balancing según el paper ---
        # Estimar y normalizar color de la luz ambiental
    ambient_color = estimate_ambient_color(flash_lin, ambient_lin, tau1=0.06, tau2=0.03)
    ambient_color = normalize_color(ambient_color)  # <- CORRECCIÓN CLAVE
    
    denoised_ambient = joint_bilateral_filter(ambient, flash, sigma_d=15, sigma_r=0.1)
    cv2.imwrite('denoised_ambient.png', denoised_ambient.astype(np.uint8))
    cv2.imshow('Denoised Ambient', denoised_ambient.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Cálculo de capa de detalle
    detail_layer = compute_detail_layer(flash, sigma_d=25, sigma_r=0.9, epsilon=0.01)
    cv2.imshow('Detail Layer', np.clip(detail_layer * 127, 0, 255).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyWindow('Detail Layer')
    cv2.imwrite('detail_layer.png', np.clip(detail_layer * 127, 0, 255).astype(np.uint8))
    
    # Detección de sombras del flash
    ambient_lin = ambient.astype(np.float32) / 255
    flash_lin = flash.astype(np.float32) / 255
    # Detectar especularidades

    specular_mask = detect_flash_specularities(flash_lin, threshold=0.95)
    cv2.imshow('Specular Mask', (specular_mask * 255).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    mask = detect_flash_shadows(flash_lin, ambient_lin, tau=0.01)
    cv2.imshow('Shadow Mask', (mask * 255).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Unir la máscara de sombras y especularidades
    full_mask = np.clip(mask + specular_mask, 0, 1)
    full_mask = cv2.GaussianBlur(full_mask, (5, 5), 5)
    full_mask = np.repeat(full_mask[..., np.newaxis], 3, axis=2)

    cv2.imshow('Full Mask', (full_mask * 255).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Fusión final
    transferred = denoised_ambient * detail_layer
    final_image = apply_masked_merge(transferred, denoised_ambient, full_mask)

    return np.clip(final_image, 0, 255).astype(np.uint8)

# Cargar imágenes
ambient = cv2.imread('img_data/flash_ambient/5_ambient.jpg').astype(np.float32)
#ambient = cv2.imread('enhanced_image_5.png').astype(np.float32)
ambient = cv2.resize(ambient, (0, 0), fx=0.3, fy=0.3)

flash = cv2.imread('img_data/flash_ambient/5_flash.jpg').astype(np.float32)
flash = cv2.resize(flash, (0, 0), fx=0.3, fy=0.3)

# Redimensionar si no están alineadas
ambient = cv2.resize(ambient, (flash.shape[1], flash.shape[0]))

# Aplicar mejora
enhanced = enhance_ambient_with_flash(ambient, flash)
cv2.imshow('Enhanced Image', enhanced)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('enhanced_image.png', enhanced)
print("Imagen mejorada guardada como 'enhanced_image.png'.")

