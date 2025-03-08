import numpy as np
import cv2

def chan_vese_level_set(image, num_iterations=100, mu=0.1, lambda1=1, lambda2=1, epsilon=1.0):
    """
    Implementación del método de Chan-Vese para la segmentación de imágenes basado en el modelo de nivel set.
    
    :param image: Imagen en escala de grises de entrada.
    :param num_iterations: Número de iteraciones para la evolución del nivel set.
    :param mu: Coeficiente de regularización para suavizar el contorno.
    :param lambda1: Coeficiente de penalización para la diferencia dentro de la región interna.
    :param lambda2: Coeficiente de penalización para la diferencia dentro de la región externa.
    :param epsilon: Parámetro para evitar divisiones por cero en el cálculo de los gradientes.
    :return: Imagen segmentada con el contorno evolucionado.
    """
    # Inicializar el contorno (nivel set) como una matriz con valores aleatorios
    phi = np.random.rand(image.shape[0], image.shape[1]) * 2 - 1  # Contorno en valores [-1, 1]
    
    # Convertir la imagen a flotante
    image = np.float64(image)
    
    # Función para calcular el promedio en la región interior y exterior del nivel set
    def mean(region):
        return np.sum(region) / np.count_nonzero(region)

    # Iteraciones para evolucionar el contorno
    for iteration in range(num_iterations):
        # Calcular los gradientes de la imagen y el nivel set
        grad_phi_x, grad_phi_y = np.gradient(phi)
        grad_phi_magnitude = np.sqrt(grad_phi_x**2 + grad_phi_y**2)
        
        # Calculamos la delta de Heaviside del nivel set
        delta_phi = np.heaviside(phi, 0)
        
        # Media de las intensidades dentro y fuera del contorno
        inside_region = image * delta_phi
        outside_region = image * (1 - delta_phi)
        
        # Promedio en la región interior y exterior
        c1 = mean(inside_region)
        c2 = mean(outside_region)
        
        # Fuerza de evolución basada en la energía de la imagen y la regularización
        force = (lambda1 * (image - c1)**2 - lambda2 * (image - c2)**2) + mu * grad_phi_magnitude
        
        # Actualizar el nivel set usando el gradiente descendente
        phi = phi - force
        
        # Regularización para mantener el nivel set suave
        phi = phi / (1 + np.abs(phi) / epsilon)
        
        # Visualización del progreso (opcional)
        if iteration % 10 == 0:
            print(f"Iteración {iteration + 1}/{num_iterations}")
    
    # Umbralizar el nivel set para obtener la segmentación final
    segmentation = np.where(phi > 0, 1, 0)
    
    return segmentation

# Cargar la imagen en escala de grises
#image = cv2.imread("img_data/astro.jpg", cv2.IMREAD_GRAYSCALE)
image = cv2.imread("img_data/Rose.jpg", cv2.IMREAD_GRAYSCALE)

image = cv2.resize(image, (800, 600))

# Aplicar el método de Chan-Vese
segmented_image = chan_vese_level_set(image, num_iterations=200)
segmented_image_uint8 = (segmented_image * 255).astype(np.uint8)

# Mostrar la imagen segmentada
cv2.imshow("Segmented Image", segmented_image_uint8)
cv2.waitKey(0)
cv2.destroyAllWindows()
