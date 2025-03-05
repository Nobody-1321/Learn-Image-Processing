from .image_pro import *
import cv2 as cv
import numpy as np

def create_channel_image(img, title):

    red, green, blue = channels_bgr(img)
    red_green, red_blue, green_blue = get_combine_channels_rg_rb_gb(img)

    if title == "blue channel":
        img = blue
    
    if title == "green channel":
        img = green

    if title == "red channel":
        img = red

    if title == "red-green channels":
        img = red_green
    
    if title == "red-blue channels":
        img = red_blue
    
    if title == "green-blue channels":
        img = green_blue

    return img

def get_flip_flop_flipflop(img, title):
    imgc = img.copy()
    flip, flop, flipflop = flip_flop_flipflop(img)

    if title == "Flip":
        imgc = flip
    
    if title == "Flop":
        imgc = flop
    
    if title == "Flip-Flop":
        imgc = flipflop

    return imgc

def get_rotate_image(img, title):
    imgc = img.copy()
    img1 = rotate_image(img, -90)
    img2 = rotate_image(imgc, 90)
    img3 = rotate_image(img, 180)

    if title == "-90":
        imgc = img1

    if title == "90":
        imgc = img2

    if title == "180":
        imgc = img3

    return imgc

def show_images_together(images, titles=None):
    """
    Muestra dos, tres o cuatro imágenes juntas en una ventana con OpenCV.

    Parámetros:
    - images: Lista de imágenes a mostrar.
    - titles: Lista de títulos para las imágenes (opcional).

    Nota: La lista de imágenes debe contener 2, 3 o 4 imágenes.
    """
    num_images = len(images)
    if num_images not in [2, 3, 4]:
        raise ValueError("La lista de imágenes debe contener 2, 3 o 4 imágenes.")

    # Determinar el tamaño de las imágenes
    height, width = images[0].shape[:2]

    # Convertir imágenes a color si son en escala de grises
    images = [cv.cvtColor(img, cv.COLOR_GRAY2BGR) if len(img.shape) == 2 else img for img in images]

    # Crear una imagen en blanco para combinar las imágenes
    if num_images == 2:
        combined_image = np.zeros((height, width * 2, 3), dtype=np.uint8)
    elif num_images == 3:
        combined_image = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)
    else:  # num_images == 4
        combined_image = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)

    # Colocar las imágenes en la imagen combinada
    combined_image[:height, :width] = images[0]
    combined_image[:height, width:width * 2] = images[1]
    if num_images > 2:
        combined_image[height:height * 2, :width] = images[2]
    if num_images == 4:
        combined_image[height:height * 2, width:width * 2] = images[3]

    # Agregar títulos si se proporcionan
    if titles:
        for i, title in enumerate(titles):
            if i == 0:
                cv.putText(combined_image, title, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
            elif i == 1:
                cv.putText(combined_image, title, (width + 10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
            elif i == 2:
                cv.putText(combined_image, title, (10, height + 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
            elif i == 3:
                cv.putText(combined_image, title, (width + 10, height + 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

    # Mostrar la imagen combinada
    cv.imshow("Combined Image", combined_image)
    cv.waitKey(0)
    cv.destroyAllWindows()