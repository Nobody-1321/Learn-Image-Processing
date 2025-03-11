import lip
import numpy as np
import cv2 as cv

image = cv.imread("img_data/caballo.webp", cv.IMREAD_GRAYSCALE)

image = np.float32(image)

# Aplicar Sobel
Gx, Gy, Gmag = lip.SobelOperator(image)

# Mostrar los resultados
cv.imshow("Gradiente en X", Gx.astype(np.uint8))
cv.imshow("Gradiente en Y", Gy.astype(np.uint8))
cv.imshow("Magnitud del Gradiente", Gmag.astype(np.uint8))
cv.waitKey(0)
cv.destroyAllWindows()