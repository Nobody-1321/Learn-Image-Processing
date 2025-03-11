import cv2
import numpy as np
import matplotlib.pyplot as plt

# Parámetros del algoritmo genético
population_size = 500
mutation_rate = 10.08
max_generations = 2000  # Máximo de generaciones
convergence_threshold = 10  # Diferencia de color mínima para detenerse

# Cargar la imagen
image_path = "img_data/rojo.webp"  # Cambia esto por tu imagen
image = cv2.imread(image_path)
image = cv2.resize(image, (400, 400))  # Ajustar tamaño para visualización
if image is None:
    raise ValueError("No se pudo cargar la imagen. Verifica la ruta.")

# Definir el área del cuadro (ajústalo según la imagen)
x, y, w, h = 100, 100, 30, 30

# Extraer la región de la imagen original como referencia
background_region = image[y:y+h, x:x+w]

# Función de aptitud mejorada: Usando el error en el espacio de color LAB para ser más perceptual
def fitness(candidate):
    # Convertir la imagen y el cuadro al espacio de color LAB para que la diferencia sea perceptualmente significativa
    candidate_lab = cv2.cvtColor(candidate, cv2.COLOR_BGR2LAB)
    background_lab = cv2.cvtColor(background_region, cv2.COLOR_BGR2LAB)
    
    # Calcular la diferencia en el espacio LAB
    diff = cv2.norm(candidate_lab, background_lab, cv2.NORM_L2)  # L2 norm
    return -diff  # Queremos minimizar esta diferencia

# Inicializar la población con cuadros de colores aleatorios
population = np.random.randint(0, 256, (population_size, h, w, 3), dtype=np.uint8)

best_fitness_values = []
converged_generation = max_generations  # Si no converge, usamos el máximo de generaciones

for gen in range(max_generations):
    # Evaluar la aptitud de cada individuo
    scores = np.array([fitness(ind) for ind in population])

    # Guardar la mejor aptitud de esta generación
    best_fitness = -np.max(scores)  # Convertimos el fitness a positivo
    best_fitness_values.append(best_fitness)

    # Seleccionar los mejores individuos
    sorted_indices = np.argsort(scores)[::-1]  # Mayor fitness primero
    selected = population[sorted_indices[:population_size // 2]]

    # Cruzamiento: Promedio de dos padres
    offspring = np.empty_like(selected)
    for i in range(len(offspring)):
        idx1, idx2 = np.random.choice(len(selected), 2, replace=False)  # Elegimos índices
        p1, p2 = selected[idx1], selected[idx2]  # Tomamos los padres
        offspring[i] = ((p1.astype(np.int32) + p2.astype(np.int32)) // 2).astype(np.uint8)

    # Mutación: Pequeñas variaciones en algunos píxeles
    mutation_mask = np.random.rand(len(offspring), h, w, 3) < mutation_rate
    mutation_values = np.random.randint(-20, 21, (len(offspring), h, w, 3), dtype=np.int32)
    offspring = np.clip(offspring.astype(np.int32) + mutation_mask * mutation_values, 0, 255).astype(np.uint8)

    # Nueva generación
    population = np.concatenate((selected, offspring))

    # Mejor candidato de la generación actual
    best_candidate = population[np.argmax(scores)]
    image_with_box = image.copy()
    image_with_box[y:y+h, x:x+w] = best_candidate

    # Mostrar evolución
    cv2.imshow("Evolucion", image_with_box)
    cv2.waitKey(1)

    # Criterio de convergencia: Si la diferencia de color promedio es menor que el umbral
    if best_fitness < convergence_threshold:
        converged_generation = gen + 1
        print(f" Convergencia alcanzada en la generación {converged_generation} con error promedio {best_fitness:.2f}")
        break

# Cálculo de métricas finales
final_diff = np.abs(best_candidate.astype(np.int32) - background_region.astype(np.int32))
error_promedio = np.mean(final_diff)
error_maximo = np.max(final_diff)
desviacion_estandar = np.std(final_diff)

print("\n Métricas de Evaluación:")
print(f" Error promedio de color (RGB): {error_promedio:.2f}")
print(f" Error máximo de color en algún píxel: {error_maximo}")
print(f" Desviación estándar del error: {desviacion_estandar:.2f}")
print(f" Generaciones hasta convergencia: {converged_generation}")

# Graficar la evolución del error
plt.figure(figsize=(10, 5))
plt.plot(best_fitness_values, label="Error Promedio", color="b")
plt.axhline(convergence_threshold, color="r", linestyle="--", label="Umbral de Convergencia")
plt.xlabel("Generaciones")
plt.ylabel("Error Promedio")
plt.title("Evolución del Error en el Algoritmo Genético")
plt.legend()
plt.grid(True)
plt.savefig("grafico_evolucion.png")  # Guardar la gráfica
plt.show()

# Mostrar el resultado final
cv2.imshow("Imagen Final", image_with_box)
cv2.imwrite("imagen_camuflada.jpg", image_with_box)
cv2.waitKey(0)
cv2.destroyAllWindows()