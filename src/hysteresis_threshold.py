import cv2
import numpy as np

def flood_fill(image, x, y, visited):
    """ Propaga la conexión de bordes fuertes a bordes débiles (8 vecinos) """
    h, w = image.shape
    stack = [(x, y)]
    
    while stack:
        px, py = stack.pop()
        
        if visited[py, px]:  # Si ya fue visitado, saltar
            continue
        
        visited[py, px] = True
        image[py, px] = 255  # Marcar como borde fuerte
        
        # Revisar vecinos en 8 direcciones
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = px + dx, py + dy
                if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx] and image[ny, nx] != 0:
                    stack.append((nx, ny))

def HysteresisThreshold(image, T_low, T_high):
    """ Aplica umbralización por histéresis """
    # Detectar bordes débiles y fuertes
    strong_edges = (image >= T_high).astype(np.uint8) * 255
    weak_edges = ((image >= T_low) & (image < T_high)).astype(np.uint8) * 255
    
    # Imagen resultado inicializada en ceros
    result = np.zeros_like(image, dtype=np.uint8)
    visited = np.zeros_like(image, dtype=np.bool_)
    
    # Propagar bordes fuertes usando flood fill
    h, w = image.shape
    for y in range(h):
        for x in range(w):
            if strong_edges[y, x] == 255 and not visited[y, x]:
                flood_fill(weak_edges, x, y, visited)  # Expande bordes fuertes
    
    return weak_edges  # Retorna la imagen binaria con los bordes confirmados

# Cargar imagen en escala de grises
#image = cv2.imread("img_data/caballo.webp", cv2.IMREAD_GRAYSCALE)
#image = cv2.imread("img_data/acat.jpg", cv2.IMREAD_GRAYSCALE)
image = cv2.imread("img_data/Rose.jpg", cv2.IMREAD_GRAYSCALE)

image = cv2.resize(image, (800, 600))

# Aplicar histéresis manualmente
final_edges = HysteresisThreshold(image, 40, 170)

# Mostrar resultado
cv2.imshow("Hysteresis Thresholding", final_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
