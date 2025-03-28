import cv2
import numpy as np
import lip
import matplotlib.pyplot as plt


'''
def sigmoid_transform(x, x_m, gamma, scale=1.0):
    """Función sigmoide modificada con factor de escala."""
    z = np.clip(-gamma * (x - x_m), -88, 88)  # Evitar valores extremos en exp
    return scale / (1 + np.exp(z))
'''

def sigmoid_transform(x, x_m, gamma):

    """Función sigmoide adaptativa con normalización."""
    x = x / 255.0  # Normalizar al rango [0, 1]
    x_m = x_m / 255.0
    z = np.clip(-gamma * (x - x_m), -88, 88)  # Evitar valores extremos en exp
    return 1 / (1 + np.exp(z)) * 255  # Desnormalizar al rango [0, 255]

def compute_optimal_gamma(Y, X_m):
    """Calcula el valor óptimo de gamma minimizando AMBE con el método de Newton-Raphson."""
    gamma = 2.0  # Valor inicial
    epsilon = 0.01  # Criterio de convergencia
    max_iter = 1000

    for _ in range(max_iter):
        # Transformación sigmoide
        transformed_Y = sigmoid_transform(Y, X_m, gamma)
        E_Y = np.mean(transformed_Y)

        error = abs(E_Y - X_m)
        
        if error < epsilon:
            break

        # Derivadas para Newton-Raphson
        dJ_dgamma = 2 * error * (E_Y - X_m)
        d2J_dgamma2 = 2 * (error ** 2)

        # Actualizar gamma
        gamma -= dJ_dgamma / max(d2J_dgamma2, 1e-6)  # Evitar divisiones por cero
    
    return np.clip(gamma, 0.4, 10)  # Limitar gamma para estabilidad
    

def apply_stretching(u, min_u, max_u, new_min, new_max):
    """Función de estiramiento de contraste."""
    alpha = (new_max - new_min) / (max_u - min_u + 1e-6)
    return new_min + alpha * (u - min_u)

def asfbhe(image):
    """Implementación del método ASFBHE."""

    # Convertir imagen a LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_cha, a_cha, b_cha = cv2.split(lab)

    # Separar los valores en dos histogramas: menores y mayores a X_m
    #X_m = np.mean(l_cha)
    X_m = lip.OtsuThreshold(l_cha)
    print(X_m)

    mask_L = l_cha <= X_m
    mask_U = l_cha > X_m
    
    histogram_L = l_cha[mask_L]  # Píxeles con luminancia menor o igual a X_m
    histogram_U = l_cha[mask_U]  # Píxeles con luminancia mayor a X_m

    # Calcular la mediana de cada subhistograma
    median_L = np.median(histogram_L)
    median_U = np.median(histogram_U)

    # Calcular gamma óptimo para cada subhistograma
    gamma_L = compute_optimal_gamma(histogram_L, median_L)
    gamma_U = compute_optimal_gamma(histogram_U, median_U)

    print(median_L, median_U)

    # Aplicar la transformación sigmoide
    u_L = median_L + (X_m - median_L) * sigmoid_transform(histogram_L, median_L, gamma_L)
    u_U = X_m + (255 - X_m) * sigmoid_transform(histogram_U, median_U, gamma_U)

    # Aplicar mapeo y estiramiento final
    l_cha_eq = np.copy(l_cha)
    l_cha_eq[mask_L] = apply_stretching(u_L, np.min(u_L), np.max(u_L), 0, X_m)
    l_cha_eq[mask_U] = apply_stretching(u_U, np.min(u_U), np.max(u_U), X_m, 255)

    # Asegurarse de que los valores de luminancia estén en el rango [0, 255]
    l_cha_eq = np.clip(l_cha_eq, 0, 255).astype(np.uint8)

    # Reconstruir la imagen
    result = cv2.merge([l_cha_eq, a_cha, b_cha])
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)  # Convertir de vuelta a BGR

    return result  # Asegurarse de devolver la imagen procesada

# Cargar imagen

#img = cv2.imread("img_data/Moises.jpg")
#img = cv2.imread("img_data/white_monta.jpg")
#img = cv2.imread("img_data/ciervo.jpg")
#img = cv2.imread("img_data/sun.jpg")
img = cv2.imread("img_data/caballo.webp")
#img = cv2.imread("img_data/input.jpg")
#img = cv2.imread("img_data/monta.jpg")
#img = cv2.imread("img_data/noct.jpeg")
#img = cv2.imread("img_data/Rose.jpg")
#img = cv2.imread("img_data/compressed.jpeg")
#img = cv.imread("img_data/sky.webp")
#img = cv2.imread("img_data/blue_dark.jpg")
#img = cv2.imread("img_data/M4.jpg")
#img = cv2.imread('img_data/astro.jpg')
#img = cv2.imread('img_data/pieces.jpg')
#img = cv2.imread("img_data/sky.webp")
#img = cv2.imread("img_data/acat_dark.png")

img = cv2.resize(img, (720, 480), interpolation=cv2.INTER_AREA)

# Validar si la imagen se cargó correctamente
if img is None:
    raise ValueError("No se pudo cargar la imagen. Verifica la ruta del archivo.")

# Aplicar ASFBHE
enhanced_img = asfbhe(img)

# Mostrar resultados
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Imagen Original")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Imagen Mejorada (ASFBHE)")
plt.imshow(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.show()
