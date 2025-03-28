# This script demonstrates the application of flood fill algorithms on a grayscale image.
# Flood fill is a technique used to fill a connected region in an image with a new color starting from a seed point.
# The script demonstrates two different flood fill algorithms: BFS (Breadth-First Search) and DFS (Depth-First Search).
# The main steps include:
# 1. Loading a grayscale image from the "img_data" directory.
# 2. Resizing the image for better display and processing.
# 3. Applying Flood Fill with BFS and DFS to different regions of the image.
# 4. Displaying the modified image after applying each flood fill algorithm.
#
# This implementation demonstrates how flood fill works with BFS and DFS in image processing.

import numpy as np
import cv2 as cv
import lip

# Load the grayscale image
img = cv.imread('img_data/drawn_image_3.png', cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (200, 200), interpolation=cv.INTER_NEAREST)

# Apply Flood Fill with BFS (Breadth-First Search)
seed_point = (5, 5)  # Seed point inside the region to be filled
new_color = 128  # New color for filling
lip.FloodFillBFS(img, seed_point, new_color)

# Display the final image after applying BFS flood fill
cv.imshow('Flood Fill BFS', img)
cv.waitKey(0)
cv.destroyAllWindows()

# Reload the image for applying DFS
img = cv.imread('img_data/drawn_image_3.png', cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (200, 200), interpolation=cv.INTER_NEAREST)

# Apply Flood Fill with DFS (Depth-First Search)
seed_point = (150, 100)  # Seed point inside the region to be filled
new_color = 128  # New color for filling
lip.FloodFillDFS(img, seed_point, new_color)

# Display the final image after applying DFS flood fill
cv.imshow('Flood Fill DFS', img)
cv.waitKey(0)
cv.destroyAllWindows()
