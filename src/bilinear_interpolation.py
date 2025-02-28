import numpy as np
import random
import cv2
import matplotlib.pyplot as plt

def generate_sparse_image(height, width, fill_percentage=0.7):
    """
    Genera una imagen con un porcentaje de píxeles aleatoriamente asignados.
    
    Parámetros:
    - height: Altura de la imagen.
    - width: Ancho de la imagen.
    - fill_percentage: Porcentaje de píxeles con color asignado (0 a 1).
    
    Retorna:
    - Imagen con píxeles parcialmente asignados.
    """
    # Crear una imagen vacía (negra)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Determinar cuántos píxeles llenar
    num_pixels = int(height * width * fill_percentage)
    
    # Asignar colores aleatorios al 70% de los píxeles
    for _ in range(num_pixels):
        y = random.randint(0, height - 1)
        x = random.randint(0, width - 1)
        image[y, x] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
    
    return image

def bilinear_interpolation(image):
    """
    Realiza interpolación bilineal para rellenar los píxeles vacíos de la imagen.
    
    Parámetros:
    - image: Imagen con píxeles parcialmente asignados (formato BGR).
    
    Retorna:
    - Imagen interpolada.
    """
    height, width, channels = image.shape
    interpolated_image = image.copy().astype(np.float32)

    # Recorrer cada canal (B, G, R)
    for c in range(channels):
        channel = interpolated_image[:, :, c]
        
        # Iterar sobre cada píxel
        for y in range(height):
            for x in range(width):
                if channel[y, x] == 0:  # Si el píxel está vacío
                    # Encontrar los píxeles vecinos más cercanos
                    neighbors = []
                    for dy in [-1, 1]:
                        for dx in [-1, 1]:
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < height and 0 <= nx < width and channel[ny, nx] != 0:
                                neighbors.append((ny, nx, channel[ny, nx]))

                    # Interpolar usando promedio ponderado (bilineal simple)
                    if neighbors:
                        weights_sum = 0
                        value_sum = 0
                        for ny, nx, value in neighbors:
                            weight = 1.0 / (abs(ny - y) + abs(nx - x))  # Peso inversamente proporcional a la distancia
                            weights_sum += weight
                            value_sum += weight * value
                        channel[y, x] = value_sum / weights_sum if weights_sum > 0 else 0

        interpolated_image[:, :, c] = channel
    
    return np.clip(interpolated_image, 0, 255).astype(np.uint8)

# Parámetros de la imagen
height, width = 50, 50  # Tamaño de la imagen
fill_percentage = 0.7  # 70% de píxeles llenos

# Generar imagen con píxeles dispersos
sparse_image = generate_sparse_image(height, width, fill_percentage)

# Aplicar interpolación bilineal
filled_image = bilinear_interpolation(sparse_image)

# Mostrar los resultados
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Imagen Original (70% de píxeles)")
plt.imshow(cv2.cvtColor(sparse_image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Imagen Interpolada")
plt.imshow(cv2.cvtColor(filled_image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.show()
