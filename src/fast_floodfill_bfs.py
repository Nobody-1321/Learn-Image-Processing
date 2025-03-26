import numpy as np
import cv2 as cv
from collections import deque  # Cola para BFS
import lip 

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
    lip.floodfill_bfs(img, seed_point, new_color)

    # Mostrar la imagen final
    cv.imshow('Flood Fill BFS', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
