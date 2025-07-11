import numpy as np
import cv2
import lip

#image = cv2.imread("img_data/habitacion.jpg", cv2.IMREAD_GRAYSCALE)
#image = cv2.imread("img_data/lena.jpg", cv2.IMREAD_GRAYSCALE)
image = cv2.imread("img_data/doom-portada.jpg", cv2.IMREAD_GRAYSCALE)

image = cv2.resize(image, (780, 480))

image = lip.BilateralFilterGrayscale(image, 1.0, 1.0, 1, 3)

edges = lip.CannyLikeDetector(image, sigma=1.0, tlow=0.19, thigh=0.29)

cv2.imshow('Canny-like Edge Detector', edges.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()