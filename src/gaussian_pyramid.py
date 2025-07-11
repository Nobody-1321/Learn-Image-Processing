import numpy as np
import cv2 as cv
import lip


image = cv.imread("img_data/lena.jpg")

# create a Gaussian pyramid with 8 levels
factor = 2 ** (1/4)  # Aproximadamente 1.1892
pyramid = lip.GaussianPyramid(image, levels=8, factor=factor)

for i, level in enumerate(pyramid):
    level = np.clip(level, 0, 255).astype(np.uint8)
    cv.imshow(f'Level {i}', level)
    cv.waitKey(0)

cv.destroyAllWindows()

