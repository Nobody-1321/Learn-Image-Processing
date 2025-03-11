import lip 
import cv2 as cv
import numpy as np

def apply_color_map(label_image):
    num_labels = np.max(label_image) + 1
    colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Fondo en negro

    color_image = colors[label_image]
    
    return color_image

def overlay_color_map(original_image, color_map):
    # Convertir la imagen original a color si es en escala de grises
    if len(original_image.shape) == 2 or original_image.shape[2] == 1:
        original_image = cv.cvtColor(original_image, cv.COLOR_GRAY2BGR)
    
    # Superponer el mapa de colores sobre la imagen original
    overlay_image = cv.addWeighted(original_image, 0.5, color_map, 0.5, 0)
    return overlay_image

if __name__ == "__main__":

    # Cargar imagen binaria (0: fondo, >0: objetos)
    #img = cv.imread("img_data/caballo.webp", cv.IMREAD_GRAYSCALE)
    #img = cv.imread("img_data/astro.jpg", cv.IMREAD_GRAYSCALE)
    img = cv.imread("img/mg.jpg", cv.IMREAD_GRAYSCALE)

    #img = cv.imread("img_data/arte.jpg", cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, (800, 800), interpolation=cv.INTER_NEAREST)

    img = cv.GaussianBlur(img, (5, 5), 0)

    # Binarizar la imagen
    _, img = cv.threshold(img, 150, 255, cv.THRESH_BINARY)
    # Invertir la imagen
    #img = cv.bitwise_not(img)

    # Aplicar Connected Components con Union-Find
    labeled_img_4 = lip.connected_components_by_union_find(img)
    labeled_img_8 = lip.connected_components_by_union_find_8_connected(img)

    # Normalizar para visualización en escala de grises (0-255)
    normalized_img_4 = cv.normalize(labeled_img_4, None, 0, 255, cv.NORM_MINMAX)
    normalized_img_8 = cv.normalize(labeled_img_8, None, 0, 255, cv.NORM_MINMAX)
    
    normalized_img_4 = np.uint8(normalized_img_4)  # Convertir a 8 bits
    normalized_img_8 = np.uint8(normalized_img_8)  # Convertir a 8 bits

    print("Número de componentes conexas:", np.max(labeled_img_4))
    print("Número de componentes conexas 8:", np.max(labeled_img_8))

    #save matrix to file
    np.savetxt("img_data/arte_4.txt", labeled_img_4, fmt='%d')
    np.savetxt("img_data/arte_8.txt", labeled_img_8, fmt='%d')

    lip.show_two_images_side_by_side(normalized_img_8, normalized_img_4, "Equitetada 8 vs. Etiquetada_4", "horizontal")

    # Aplicar color map    
    color_img_4 = apply_color_map(labeled_img_4)
    color_img_8 = apply_color_map(labeled_img_8)

    # Superponer el mapa de colores sobre la imagen original
    overlay_img_4 = overlay_color_map(img, color_img_4)
    overlay_img_8 = overlay_color_map(img, color_img_8)
    
    #gaussian blur
    overlay_img_4 = cv.GaussianBlur(overlay_img_4, (5, 5), 0)
    overlay_img_8 = cv.GaussianBlur(overlay_img_8, (5, 5), 0)

    kernel = np.ones((3,3), np.uint8)
    overlay_img_4 = cv.morphologyEx(overlay_img_4, cv.MORPH_CLOSE, kernel)  # Eliminación de ruido
    overlay_img_8 = cv.morphologyEx(overlay_img_8, cv.MORPH_CLOSE, kernel)  # Eliminación de ruido
    
    # Aplicar erosión para mejorar la visualización
    overlay_img_4 = cv.erode(overlay_img_4, kernel, iterations=1)
    overlay_img_8 = cv.erode(overlay_img_8, kernel, iterations=1)

    # Mostrar las imágenes etiquetadas en color superpuestas sobre la imagen original
    lip.show_two_images_side_by_side(normalized_img_4, overlay_img_4, "Etiquetada 4", "horizontal")
    lip.show_two_images_side_by_side(normalized_img_8, overlay_img_8, "Etiquetada 8", "horizontal")
    cv.waitKey(0)
    cv.destroyAllWindows()