import cv2

# Cargar la imagen en color
imagen = cv2.imread("img_data/lena.jpg")  # Reemplaza con el nombre de tu imagen

# Convertir a escala de grises
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Guardar la imagen en escala de grises
cv2.imwrite("img_data/lena_gray.jpg", imagen_gris)

# Mostrar las imágenes
cv2.imshow("Imagen en Color", imagen)
cv2.imshow("Imagen en Blanco y Negro", imagen_gris)

# Esperar una tecla para cerrar
cv2.waitKey(0)
cv2.destroyAllWindows()
