
''''
import cv2
import numpy as np

def external_energy(image, point):
    """Calcula la energía externa en un punto basado en el gradiente."""
    grad = cv2.Laplacian(image, cv2.CV_64F)  # Bordes con Laplaciano
    x, y = int(point[0]), int(point[1])
    return np.abs(grad[y, x]) if 0 <= x < image.shape[1] and 0 <= y < image.shape[0] else np.inf

def snake_open_iter(image, contour, alpha=1.0):
    """Ejecuta UNA iteración del algoritmo Snake para contornos abiertos."""
    image = cv2.GaussianBlur(image, (5, 5), 0)  # Suavizar imagen
    n = len(contour)  # Número de vértices

    # Matriz de costos F y matriz de seguimiento P
    F = np.full((n, 9), np.inf)  # Cada vértice tiene 9 posibles movimientos
    P = np.zeros((n, 9), dtype=int)  # Para almacenar el mejor movimiento previo

    # 1️⃣ **Inicializar la primera fila de F usando solo energía externa**
    for j in range(9):
        dx, dy = j % 3 - 1, j // 3 - 1  # Desplazamientos (-1, 0, 1) en x e y
        new_point = contour[0] + np.array([dx, dy])
        F[0, j] = external_energy(image, new_point)

    # 2️⃣ **Llenar el resto de la matriz F**
    for i in range(1, n):
        for j in range(9):
            dx, dy = j % 3 - 1, j // 3 - 1  # Movimiento actual
            new_point = contour[i] + np.array([dx, dy])

            min_energy = np.inf
            best_j = 0

            for jr in range(9):  # Buscar la mejor transición desde la fila anterior
                prev_dx, prev_dy = jr % 3 - 1, jr // 3 - 1
                prev_point = contour[i - 1] + np.array([prev_dx, prev_dy])
                energy = F[i - 1, jr] + alpha * np.linalg.norm(prev_point - new_point)

                if energy < min_energy:
                    min_energy = energy
                    best_j = jr  # Guardar mejor transición

            F[i, j] = external_energy(image, new_point) + min_energy
            P[i, j] = best_j  # Guardar de dónde vino

    # 3️⃣ **Encontrar la última posición con menor energía**
    min_j = np.argmin(F[-1])  # Índice con menor energía en la última fila
    new_contour = [contour[-1] + np.array([min_j % 3 - 1, min_j // 3 - 1])]

    # 4️⃣ **Recorrer P para reconstruir el mejor camino**
    for i in range(n - 2, -1, -1):
        min_j = P[i + 1, min_j]
        new_contour.append(contour[i] + np.array([min_j % 3 - 1, min_j // 3 - 1]))

    return np.array(new_contour[::-1])  # Devolver el contorno refinado

def snake_closed(image, contour, max_iterations=100, tolerance=0.1):
    """
    Ejecuta el algoritmo Snake para contornos cerrados y muestra las iteraciones en OpenCV.
    """
    prev_contour = contour.copy()
    
    for i in range(max_iterations):
        contour = snake_open_iter(image, contour)  # Aplicar una iteración
        
        # Cerrar el contorno asegurando que el primer y último punto coincidan
        contour[-1] = contour[0]
        
        # Visualizar la iteración en OpenCV
        img_display = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for j in range(len(contour) - 1):
            cv2.line(img_display, tuple(contour[j].astype(int)), tuple(contour[j + 1].astype(int)), (0, 0, 255), 1)
        cv2.line(img_display, tuple(contour[-1].astype(int)), tuple(contour[0].astype(int)), (0, 255, 0), 1)

        cv2.imshow('Snake Algorithm - Iteration', img_display)
        cv2.waitKey(50)  # Pequeño delay para ver el proceso en tiempo real

        # Verificar convergencia
        if np.allclose(contour, prev_contour, atol=tolerance):
            break
        prev_contour = contour.copy()
    
    cv2.destroyAllWindows()
    return contour

# Cargar imagen en escala de grises
image = cv2.imread('img_data/drawn_image_4.png', cv2.IMREAD_GRAYSCALE)
h, w = image.shape

# Generar un contorno inicial (círculo)
t = np.linspace(0, 2 * np.pi, 150)
r = 100
cx, cy = 100, 140
contour_init = np.array([(cx + r * np.cos(a), cy + r * np.sin(a)) for a in t])

# Aplicar el algoritmo Snake-Closed con visualización en tiempo real
contour_final = snake_closed(image, contour_init, max_iterations=100, tolerance=0.2)

# Mostrar el resultado final
cv2.imshow("Final Contour", cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
for j in range(len(contour_final) - 1):
    cv2.line(image, tuple(contour_final[j].astype(int)), tuple(contour_final[j + 1].astype(int)), (255, 0, 0), 1)
cv2.line(image, tuple(contour_final[-1].astype(int)), tuple(contour_final[0].astype(int)), (255, 0, 0), 1)

cv2.imshow('Final Contour', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

def initialize_phi(shape, radius=250):
    """Inicializa la función de nivel Φ como una distancia firmada desde un círculo."""
    h, w = shape
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    cx, cy = w // 2, h // 2  # Centro del círculo
    return np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - radius  # Distancia firmada

def compute_divergence(F):
    """Calcula la divergencia del gradiente normalizado de F."""
    F_x = np.gradient(F, axis=1)
    F_y = np.gradient(F, axis=0)
    norm_grad = np.sqrt(F_x**2 + F_y**2 + 1e-10)

    F_x /= norm_grad
    F_y /= norm_grad

    div_F = np.gradient(F_x, axis=1) + np.gradient(F_y, axis=0)
    return div_F

def chan_vese(I, max_iters=50000, tol=0.1, lambda1=20, lambda2=0, mu=0.0, dt=0.9):
    """
    Implementa el algoritmo Chan-Vese con el flujo correcto.
    
    1. Se computa la divergencia del gradiente normalizado de F.
    2. Se calcula DF para actualizar F.
    3. Se suma DF a F de la iteración anterior.
    4. Se reinicializa F.
    
    Parámetros:
        I: Imagen en escala de grises.
        max_iters: Número máximo de iteraciones.
        tol: Umbral de convergencia.
        lambda1, lambda2: Pesos de segmentación.
        mu: Peso del término de curvatura.
        dt: Paso de tiempo para actualización de Φ.
    
    Retorna:
        Máscara binaria de la región segmentada.
    """
    I = I.astype(np.float32)
    I = (I - I.min()) / (I.max() - I.min())  # Normalizar imagen entre [0,1]
    
    phi = initialize_phi(I.shape)  # Función de nivel inicial
    prev_phi = np.copy(phi)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    for i in range(max_iters):
        # Calcular regiones dentro y fuera del contorno
        inside = I[phi > 0]
        outside = I[phi <= 0]
        
        # Medias dentro y fuera
        c1 = np.mean(inside) if len(inside) > 0 else 0
        c2 = np.mean(outside) if len(outside) > 0 else 0

        # Calcular la divergencia del gradiente normalizado
        div_F = compute_divergence(phi)

        # Calcular el término de evolución DF
        dPhi = lambda1 * (I - c1)**2 - lambda2 * (I - c2)**2 + mu * div_F

        # Actualizar F
        phi += dt * dPhi / (1 + np.abs(dPhi))

        # Re-inicialización usando distancia firmada cada 10 iteraciones
        if i % 10 == 0:
            phi = distance_transform_edt(phi > 0) - distance_transform_edt(phi <= 0)

        # Verificar convergencia
        speed = np.max(np.abs(phi - prev_phi))
        prev_phi = np.copy(phi)

        # Mostrar evolución cada 20 iteraciones
        if i % 20 == 0:
            ax[0].imshow(I, cmap='gray')
            ax[0].set_title("Imagen Original")
            ax[1].cla()
            ax[1].imshow(phi > 0, cmap='gray')
            ax[1].set_title(f"Iteración {i}")
            plt.pause(0.1)

        if speed < tol:
            print(f"Convergió en la iteración {i}")
            break
    
    plt.show()
    return phi > 0  # Retorna la máscara binaria segmentada

# Cargar imagen en escala de grises
#image = cv2.imread('img_data/drawn_image_4.png', cv2.IMREAD_GRAYSCALE)
image = cv2.imread('img_data/acat.jpg', cv2.IMREAD_GRAYSCALE)
#image = cv2.imread('img_data/caballo.webp', cv2.IMREAD_GRAYSCALE)

image = cv2.resize(image, (500, 500))  # Redimensionar imagen

#image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)[1]  # Binarizar imagen

# Invertir blanco a negro y negro a blanco (opcional)

# Aplicar el algoritmo Chan-Vese
segmented_mask = chan_vese(image)
