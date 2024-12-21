import cv2
import numpy as np

def create_binary_mask(image, lower_bound, upper_bound):
    """
    Genera una máscara binaria basada en un rango de color.
    
    Parámetros:
    - image: Imagen en formato BGR.
    - lower_bound: Límite inferior del color en formato HSV.
    - upper_bound: Límite superior del color en formato HSV.
    
    Retorna:
    - Máscara binaria.
    """
    # Verificar que la imagen no sea None
    if image is None:
        raise ValueError("La imagen no se pudo cargar. Verifica la ruta del archivo.")

    # Convertir la imagen a HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Crear la máscara para los píxeles dentro del rango
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    
    return mask

# Cargar la imagen original y verificar que se cargó correctamente
image = cv2.imread("img_data/img2.jpeg")
if image is None:
    raise FileNotFoundError("No se pudo cargar la imagen. Verifica la ruta del archivo.")

# Cargar la imagen de fondo
background = cv2.imread("img_data/img11.jpg")
if background is None:
    raise FileNotFoundError("No se pudo cargar la imagen de fondo. Verifica la ruta del archivo.")

# Redimensionar el fondo para que coincida con las dimensiones de la imagen original
background = cv2.resize(background, (image.shape[1], image.shape[0]))

# Definir límites para el color verde en HSV
lower_green = np.array([35, 50, 50])  # Límite inferior
upper_green = np.array([85, 255, 255])  # Límite superior

# Generar la máscara para el color verde
mask = create_binary_mask(image, lower_green, upper_green)

# Crear una máscara invertida
inverse_mask = cv2.bitwise_not(mask)

# Extraer la persona  del primer plano (imagen original) sin el fondo verde
person = cv2.bitwise_and(image, image, mask=inverse_mask)

# poner la persona en el fondo
background = cv2.bitwise_and(background, background, mask=mask)

# Sumar las dos imágenes
result = cv2.add(person, background)

# Guardar y mostrar los resultados
cv2.imshow("Mascara", mask)
cv2.imshow("Resultado", person)
cv2.imshow("Fondo", background)
cv2.imshow("Resultado final", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
