import cv2 as cv
import numpy as np

def channels_bgr(img):

    b, g, r = cv.split(img)
    zeros = np.zeros(img.shape[:2], dtype="uint8")

    blue_img = cv.merge([b, zeros, zeros])
    green_img = cv.merge([zeros, g, zeros])
    red_img = cv.merge([zeros, zeros, r])

    return blue_img, green_img, red_img

def channels_hsv(img):
        
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        zeros = np.zeros(img.shape[:2], dtype="uint8")
        
        hue_img = cv.merge([h, zeros, zeros])
        saturation_img = cv.merge([zeros, s, zeros])
        value_img = cv.merge([zeros, zeros, v])
        
        return hue_img, saturation_img, value_img

def open_image(path):
    img = cv.imread(path)
    return img
    
def resize_image(img, width, height):
    return cv.resize(img, (width, height), interpolation=cv.INTER_AREA)

def save(filename, img):
    try:
        cv.imwrite(filename, img)
    except Exception as e:
        print(f"Error saving image: {e}")   

def combine_channels(img, red, green, blue):
    
    if red.shape[:2] != img.shape[:2] or green.shape[:2] != img.shape[:2] or blue.shape[:2] != img.shape[:2]:
        print("Error: channel images must have the same dimensions as the original image")
        return None
        
    return cv.merge([blue, green, red])

def flip_flop_flipflop(img):
    imgc = img.copy()
    return cv.flip(img, -1), cv.flip(imgc, 1), cv.flip(img, -1)

def rotate_image(img, angle):
    rows, cols = img.shape[:2]
    M = cv.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    return cv.warpAffine(img, M, (cols, rows))

def get_combine_channels_rg_rb_gb(img):
    blue, green, red = cv.split(img)
    zeros = np.zeros(img.shape[:2], dtype="uint8")
    
    red_green = cv.merge([zeros, green, red])
    red_blue = cv.merge([blue, zeros, red])
    green_blue = cv.merge([blue, green, zeros])
    
    return red_green, red_blue, green_blue

def histogram_equalization(img):
    # 1. Calcular el histograma
    hist, bins = np.histogram(img.flatten(), bins=256, range=[0, 256])
    
    # 2. Calcular la función de distribución acumulativa (CDF)
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]  # Normalizar el CDF para que vaya de 0 a 1
    
    # 3. Mapear los valores originales a los nuevos valores usando la CDF
    cdf_scaled = (cdf_normalized * 255).astype(np.uint8)  # Escalar a [0, 255]
    img_equalized = cdf_scaled[img]  # Usar la CDF como lookup table
    
    return img_equalized

def histogram_matching(img_1, img_2):
    # 1. Calcular el histograma de la imagen de referencia
    hist_ref, bins = np.histogram(img_1.flatten(), bins=256, range=[0, 256])
    
    # 2. Calcular el histograma de la imagen a ecualizar
    hist, bins = np.histogram(img_2.flatten(), bins=256, range=[0, 256])
    
    # 3. Calcular la función de distribución acumulativa (CDF) de ambas imágenes
    cdf_ref = hist_ref.cumsum()
    cdf_ref_normalized = cdf_ref / cdf_ref[-1]
    
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]
    
    # 4. Crear la tabla de correspondencia
    lookup_table = np.interp(cdf_normalized, cdf_ref_normalized, range(256)).astype(np.uint8)
    
    # 5. Aplicar la tabla de correspondencia a la imagen a ecualizar
    img_matched = lookup_table[img_2]
    
    return img_matched

def bgr_to_gray(img):
    blue, green, red = cv.split(img)
    gray_img = 0.299 * red + 0.587 * green + 0.114 * blue
    return gray_img.astype(np.uint8)
    
