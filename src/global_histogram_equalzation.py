# This script demonstrates the application of histogram equalization on both grayscale and color images.
# The main steps include:
# 1. Loading an image from the "img_data" directory.
# 2. Resizing the image to a consistent resolution (600x600).
# 3. Applying histogram equalization on grayscale images to enhance contrast.
# 4. Applying histogram equalization on color images (RGB) to improve overall image quality.
# 5. Displaying the original image and the histogram-equalized image along with their histograms.
# 
# This implementation demonstrates how histogram equalization can improve image contrast in both grayscale and color images.

import cv2 as cv
import numpy as np
import lip

# Load a grayscale image from the "img_data" directory
# Uncomment the desired image to test
# img = cv.imread("img_data/Moises.jpg", cv.IMREAD_GRAYSCALE)
# img = cv.imread("img_data/white_monta.jpg", cv.IMREAD_GRAYSCALE)
# img = cv.imread("img_data/ciervo.jpg", cv.IMREAD_GRAYSCALE)
# img = cv.imread("img_data/noct.jpeg", cv.IMREAD_GRAYSCALE)
# img = cv.imread("img_data/Rose.jpg", cv.IMREAD_GRAYSCALE)
# img = cv.imread("img_data/compressed.jpeg", cv.IMREAD_GRAYSCALE)
img = cv.imread("img_data/sky.webp", cv.IMREAD_GRAYSCALE)

# Resize the grayscale image to a fixed resolution (600x600)
img = cv.resize(img, (600, 600), interpolation=cv.INTER_AREA)

# Apply Histogram Equalization to enhance contrast globally on the grayscale image
img_eq_lip = lip.HistogramEqualizationGray(img)

# Display the original grayscale image and the histogram-equalized image with histograms
lip.plot_two_images_and_histograms(img, img_eq_lip, "Original", "img_eq_lip")

# Load a color image from the "img_data" directory
# Uncomment the desired image to test
# img = cv.imread("img_data/Moises.jpg")
# img = cv.imread("img_data/white_monta.jpg")
# img = cv.imread("img_data/ciervo.jpg")
# img = cv.imread("img_data/noct.jpeg")
# img = cv.imread("img_data/Rose.jpg")
# img = cv.imread("img_data/compressed.jpeg")
img = cv.imread("img_data/sky.webp")

# Resize the color image to a fixed resolution (600x600)
img = cv.resize(img, (600, 600), interpolation=cv.INTER_AREA)

# Apply Histogram Equalization to enhance contrast globally on the color image (RGB)
img_eq_lip = lip.HistogramEqualizationRGB(img)

# Display the original color image and the histogram-equalized image with histograms
lip.plot_two_images_and_histograms(img, img_eq_lip, "Original", "img_eq_lip")
