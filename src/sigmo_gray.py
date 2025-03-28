import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def asfbhe_grayscale(image):
    """Implementación del método ASFBHE para imágenes en escala de grises."""

    # Validar que la imagen sea en escala de grises
    if len(image.shape) != 2:
        raise ValueError("La imagen debe estar en escala de grises.")

    # Separar los valores en dos histogramas: menores y mayores a X_m
    X_m = np.mean(image)  # Usar la media como umbral
    print(f"Umbral calculado (X_m): {X_m}")

    mask_L = image <= X_m
    mask_U = image > X_m
    
    histogram_L = image[mask_L]  # Píxeles con luminancia menor o igual a X_m
    histogram_U = image[mask_U]  # Píxeles con luminancia mayor a X_m

    # Calcular la mediana de cada subhistograma
    median_L = np.median(histogram_L)
    median_U = np.median(histogram_U)

    # Calcular gamma óptimo para cada subhistograma
    gamma_L = compute_optimal_gamma(histogram_L, median_L)
    gamma_U = compute_optimal_gamma(histogram_U, median_U)

    print(f"Mediana de la región baja (median_L): {median_L}")
    print(f"Mediana de la región alta (median_U): {median_U}")

    # Aplicar la transformación sigmoide
    u_L = median_L + (X_m - median_L) * sigmoid_transform(histogram_L, median_L, gamma_L)
    u_U = X_m + (255 - X_m) * sigmoid_transform(histogram_U, median_U, gamma_U)

    # Aplicar mapeo y estiramiento final
    image_eq = np.copy(image)
    image_eq[mask_L] = apply_stretching(u_L, np.min(u_L), np.max(u_L), 0, X_m)
    image_eq[mask_U] = apply_stretching(u_U, np.min(u_U), np.max(u_U), X_m, 255)

    # Asegurarse de que los valores de luminancia estén en el rango [0, 255]
    image_eq = np.clip(image_eq, 0, 255).astype(np.uint8)

    return image_eq  # Devolver la imagen procesada

# Cargar imagen en escala de grises
#img = cv2.imread("img_data/Moises.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.imread("img_data/acat.jpg", cv2.IMREAD_GRAYSCALE)


# Validar si la imagen se cargó correctamente
if img is None:
    raise ValueError("No se pudo cargar la imagen. Verifica la ruta del archivo.")

# Aplicar ASFBHE
enhanced_img = asfbhe_grayscale(img)

# Mostrar resultados
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Imagen Original")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Imagen Mejorada (ASFBHE)")
plt.imshow(enhanced_img, cmap="gray")
plt.axis("off")

plt.show()