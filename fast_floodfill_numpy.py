import numpy as np
import cv2 as cv

def floodfill_numpy(I, p, new_color):
    orig_color = I[p[1], p[0]]
    if orig_color == new_color:
        return
    
    mask = I == orig_color  # Máscara de la región a rellenar
    queue = [p]

    while queue:
        x, y = queue.pop(0)
        if mask[y, x]:
            I[y, x] = new_color
            mask[y, x] = False  # Marca como procesado
            
            # Agrega vecinos
            for qx, qy in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
                if 0 <= qx < I.shape[1] and 0 <= qy < I.shape[0]:
                    queue.append((qx, qy))
                    cv.imshow('Flood Fill BFS', I)
                    cv.waitKey(1)                

# Uso
img = cv.imread('img_data/drawn_image_3.png', cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (100, 100), interpolation=cv.INTER_NEAREST)
floodfill_numpy(img, (5, 5), 128)
cv.imshow('Flood Fill Numpy', img)
cv.waitKey(0)
cv.destroyAllWindows()
