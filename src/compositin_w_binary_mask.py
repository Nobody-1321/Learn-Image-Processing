import cv2
import numpy as np

def compositing_with_binary_mask(background, foreground, mask):
    """
    Combina una imagen de fondo y una de primer plano usando una máscara binaria.
    
    Parámetros:
    - background: Imagen de fondo (numpy array).
    - foreground: Imagen de primer plano (numpy array).
    - mask: Máscara binaria (numpy array con valores 0 y 1 o 0 y 255).
    
    Retorna:
    - Imagen compuesta.
    """
    
    # Asegurarse de que la máscara sea binaria (0 o 1)
    binary_mask = mask / 255 if mask.max() > 1 else mask
    
    binary_mask = np.expand_dims(mask, axis=-1)

    # Asegurarse de que las dimensiones coincidan
    if background.shape != foreground.shape:
        raise ValueError("Las dimensiones del fondo, el primer plano y la máscara deben coincidir.")
    
    # Composición
    composed_image = (binary_mask * foreground + (1 - binary_mask) * background).astype(np.uint8)
    return composed_image

# Cargar las imágenes y la máscara
background = cv2.imread("img_data/img2.jpeg")
foreground = cv2.imread("img_data/img11.jpg")
mask = cv2.imread("binary_mask.jpg", cv2.IMREAD_GRAYSCALE)

# Redimensionar para que coincidan
background = cv2.resize(background, (foreground.shape[1], foreground.shape[0]))
mask = cv2.resize(mask, (foreground.shape[1], foreground.shape[0]))

# Aplicar la composición
result = compositing_with_binary_mask(background, foreground, mask)

# Mostrar y guardar el resultado
cv2.imshow("Resultado", result)
cv2.imwrite("composited_image.jpg", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
