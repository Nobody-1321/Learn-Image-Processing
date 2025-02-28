import ena 
import cv2 as cv
import numpy as np

if __name__ == "__main__":
    # Cargar imagen binaria (0: fondo, >0: objetos)
    img = cv.imread("img_data/img2.jpeg", cv.IMREAD_GRAYSCALE)
    #img = cv.imread("img/YU.jpg", cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, (800, 800), interpolation=cv.INTER_NEAREST)

    # Aplicar Connected Components con Union-Find
    #labeled_img = ena.connected_components_by_union_find(img)
    labeled_img = ena.connected_components_by_union_find_8_connected(img)

    # Normalizar para visualización en escala de grises (0-255)
    normalized_img = cv.normalize(labeled_img, None, 0, 255, cv.NORM_MINMAX)
    normalized_img = np.uint8(normalized_img)  # Convertir a 8 bits

    print("Número de componentes conexas:", np.max(labeled_img))
    print(labeled_img)

    ena.show_two_images_side_by_side(img, normalized_img, "Original vs. Etiquetada", "horizontal")