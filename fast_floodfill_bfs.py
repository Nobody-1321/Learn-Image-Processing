import numpy as np
import cv2 as cv
from collections import deque  # Cola para BFS

def floodfill_bfs(I, p, new_color):
    orig_color = I[p[1], p[0]]
    if orig_color == new_color:
        return

    queue = deque([p])  # Cola FIFO
    I[p[1], p[0]] = new_color

    while queue:
        x, y = queue.popleft()  # Extrae el primer elemento (FIFO)
        
        # Explora los vecinos (derecha, izquierda, abajo, arriba)
        for q in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
            qx, qy = q
            if 0 <= qx < I.shape[1] and 0 <= qy < I.shape[0] and I[qy, qx] == orig_color:
                queue.append((qx, qy))
                I[qy, qx] = new_color

                # Visualización del proceso
                cv.imshow('Flood Fill BFS', I)
                cv.waitKey(1)

# Uso de la función
if __name__ == "__main__":
    # Cargar imagen en escala de grises
    img = cv.imread('img_data/drawn_image_3.png', cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, (100, 100), interpolation=cv.INTER_NEAREST)

    print("Imagen original:")
    print(img)

    # Aplicar Flood Fill con BFS
    seed_point = (5, 5)  # Punto inicial dentro de la región a llenar
    new_color = 128  # Nuevo color
    floodfill_bfs(img, seed_point, new_color)

    print("Imagen después del Flood Fill:")
    print(img)

    # Mostrar la imagen final
    cv.imshow('Flood Fill BFS', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
