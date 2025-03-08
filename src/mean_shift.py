import numpy as np
import cv2

def gaussian_weight(distance, bandwidth):
    """Calcula el peso Gaussiano."""
    return np.exp(-distance / (2 * bandwidth ** 2))

def MeanShiftFilterFast(I, hs, hr, grid_step=2, max_iter=10, threshold=1e-3):
    """
    Filtro Mean-Shift optimizado con submuestreo en una malla.
    
    Parámetros:
    - I: Imagen en escala de grises (numpy array).
    - hs: Parámetro de ancho de banda espacial.
    - hr: Parámetro de ancho de banda de intensidad.
    - grid_step: Tamaño del paso en la malla de submuestreo.
    - max_iter: Máximo de iteraciones por píxel.
    - threshold: Umbral de convergencia.

    Retorna:
    - Imagen filtrada.
    """
    I = I.astype(np.float32) / 255.0  # Normalizar imagen [0,1]
    height, width = I.shape
    Ir = np.zeros_like(I)  # Imagen de salida

    for y in range(0, height, grid_step):
        for x in range(0, width, grid_step):
            # Inicializar xr, yr, vr en el píxel actual
            xr, yr, vr = x, y, I[y, x]

            for _ in range(max_iter):
                num_x, num_y, num_v = 0.0, 0.0, 0.0
                den = 0.0

                # Buscar vecinos en una ventana reducida (submuestreo)
                for yi in range(max(0, y - hs), min(height, y + hs + 1), grid_step):
                    for xi in range(max(0, x - hs), min(width, x + hs + 1), grid_step):
                        vi = I[yi, xi]

                        # Calcular peso
                        spatial_dist = (xr - xi) ** 2 + (yr - yi) ** 2
                        range_dist = (vr - vi) ** 2
                        w = gaussian_weight(spatial_dist, hs) * gaussian_weight(range_dist, hr)

                        # Acumular valores ponderados
                        num_x += w * xi
                        num_y += w * yi
                        num_v += w * vi
                        den += w

                # Actualizar nueva posición y color
                new_xr, new_yr, new_vr = num_x / den, num_y / den, num_v / den
                mean_shift = np.sqrt((new_xr - xr) ** 2 + (new_yr - yr) ** 2 + (new_vr - vr) ** 2)

                # Si la variación es menor al umbral, detener iteración
                if mean_shift < threshold:
                    break

                xr, yr, vr = new_xr, new_yr, new_vr

            # Asignar nuevo valor al píxel
            Ir[y:y+grid_step, x:x+grid_step] = vr  # Asignar en bloques

    return (Ir * 255).astype(np.uint8)  # Convertir a uint8 (0-255)

# Ejemplo de uso
def main():
    #img = cv2.imread("img_data/caballo.webp", cv2.IMREAD_GRAYSCALE)
    #img = cv2.imread("img_data/astro.jpg", cv2.IMREAD_GRAYSCALE)
    #img = cv2.imread("img_data/caballo.webp", cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (400, 400))

    # Aplicar Mean-Shift Filtering Optimizado
    hs = 32  # Ancho de banda espacial
    hr = 16  # Ancho de banda de color
    img_filtered = MeanShiftFilterFast(img, hs, hr, grid_step=4)  # Submuestreo cada 4 píxeles

    cv2.imshow("Original", img)
    cv2.imshow("Filtrada (Mean-Shift Rapido)", img_filtered)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
