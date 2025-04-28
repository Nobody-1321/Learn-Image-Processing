import cv2
import numpy as np

# Cargar la imagen original
image_path = './img_data/lena.jpg'  # <-- cambia esta ruta
image = cv2.imread(image_path)
# escalar la imagen a un tamaño más pequeño
image = cv2.resize(image, (720, 720))

if image is None:
    raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")

# Crear la máscara (negra al inicio)
mask = np.zeros(image.shape[:2], dtype=np.uint8)

# Parámetros del pincel
brush_radius = 15
drawing = False

# Función para manejar los eventos del mouse
def draw_mask(event, x, y, flags, param):
    global drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(mask, (x, y), brush_radius, 255, -1)  # Dibujar en la máscara
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

# Crear una ventana y asignar la función de mouse
cv2.namedWindow('Dibujar Mascara')
cv2.setMouseCallback('Dibujar Mascara', draw_mask)

while True:
    # Mostrar imagen combinada para ver la máscara en rojo sobre la imagen
    mask_colored = cv2.merge([mask, np.zeros_like(mask), np.zeros_like(mask)])  # Solo canal rojo
    overlay = cv2.addWeighted(image, 0.7, mask_colored, 0.3, 0)
    cv2.imshow('Dibujar Mascara', overlay)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        # Guardar la máscara
        cv2.imwrite('mascara_guardada.png', mask)
        print('Máscara guardada como "mascara_guardada.png".')
    elif key == 27:  # ESC para salir
        break

cv2.destroyAllWindows()
