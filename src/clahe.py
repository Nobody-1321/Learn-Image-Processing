# This script demonstrates the application of histogram equalization and CLAHE (Contrast Limited Adaptive Histogram Equalization)
# on both grayscale and color images. The main steps include:
# 1. Loading an image from the "img_data" directory.
# 2. Resizing the image to a consistent resolution (720x480).
# 3. Applying global Histogram Equalization to enhance image contrast.
# 4. Applying CLAHE, which enhances contrast locally and prevents over-amplification of noise.
# 5. Displaying the original image, the histogram-equalized image, and the CLAHE-processed image along with their histograms.
# 
# The script first applies these operations to a grayscale image, then to a color image, showing the differences in the results.
# This demonstrates how global and local contrast enhancement techniques can be applied to improve image quality.

import numpy as np
import cv2 as cv
import lip

# Load a grayscale image from the "img_data" directory
# Uncomment the desired image to test
# img = cv.imread("img_data/Moises.jpg", cv.IMREAD_GRAYSCALE)
# img = cv.imread("img_data/white_monta.jpg", cv.IMREAD_GRAYSCALE)
# img = cv.imread("img_data/escena.jpg", cv.IMREAD_GRAYSCALE)
# img = cv.imread("img_data/acat.jpg", cv.IMREAD_GRAYSCALE)
# img = cv.imread("img_data/ciervo.jpg", cv.IMREAD_GRAYSCALE)
img = cv.imread("img_data/noct.jpeg", cv.IMREAD_GRAYSCALE)
# img = cv.imread("img_data/Rose.jpg", cv.IMREAD_GRAYSCALE)

# Resize the image to a fixed resolution (720x480) for consistency
img = cv.resize(img, (720, 480), interpolation=cv.INTER_AREA)

# Apply Histogram Equalization (global contrast enhancement)
img_equ = lip.HistogramEqualization(img)

# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
# CLAHE enhances contrast locally in small regions while preventing over-amplification
img_clahe = lip.HistogramEqualizationClaheGrayscale(img, clip_limit=100, grid_size=(9, 9))

# Display the original image, histogram-equalized image, and CLAHE-processed image with histograms
lip.plot_three_images_and_histograms(img, img_equ, img_clahe, "Original", "Histogram Equalization", "CLAHE")

# Load a color image from the "img_data" directory
# Uncomment the desired image to test
# img = cv.imread("img_data/Moises.jpg")
# img = cv.imread("img_data/white_monta.jpg")
img = cv.imread("img_data/ciervo.jpg")
# img = cv.imread("img_data/sun.jpg")
# img = cv.imread("img_data/caballo.webp")
# img = cv.imread("img_data/input.jpg")
# img = cv.imread("img_data/monta.jpg")
# img = cv.imread("img_data/noct.jpeg")
# img = cv.imread("img_data/Rose.jpg")
# img = cv.imread("img_data/compressed.jpeg")
# img = cv.imread("img_data/sky.webp")
# img = cv.imread("img_data/blue_dark.jpg")
# img = cv.imread("img_data/M4.jpg")
# img = cv.imread('img_data/astro.jpg')

# Resize the color image to a fixed resolution (720x480)
img = cv.resize(img, (720, 480), interpolation=cv.INTER_AREA)

# Apply Histogram Equalization to enhance contrast globally
img_equa = lip.HistogramEqualization(img)

# Apply CLAHE to the color image (performs local contrast enhancement)
img_clahe = lip.HistogramEqualizationClaheRGB(img, clip_limit=17, grid_size=(9, 9))

# Display the original image, histogram-equalized image, and CLAHE-processed image with histograms
lip.plot_three_images_and_histograms(img, img_equa, img_clahe, "Original", "Histogram Equalization", "CLAHE")
