import numpy as np
import cv2
import matplotlib.pyplot as plt
import lip 

img = cv2.imread('img_data/mappertres.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (512, 512))

mag, phase = lip.ComputeFourierSpectra(img)

combined = np.hstack((img, mag ))

cv2.imshow("Original Magnitude", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()

notch_coords = [(0, 14), (10, 11)]

d0 = 2
n = 2

img_filtered, spectrum, notch_mask = lip.NotchFiltering(img, d0, notch_coords, n)
mag_filtered, phase_filtered = lip.ComputeFourierSpectra(img_filtered)

img_filtered = cv2.normalize(img_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


combined_filtered = np.hstack((img, img_filtered, mag_filtered))
cv2.imshow("Original | Filtered | Filtered Magnitude", combined_filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()