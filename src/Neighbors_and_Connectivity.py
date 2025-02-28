import numpy as np
import cv2 as cv

def create_matrix(rows, cols):
    """Create a matrix with given rows and columns filled with random integers."""
    return np.random.randint(0, 2, size=(rows, cols), dtype=np.uint8)

def get_neighbors(matrix, row, col):
    """Get the neighbors of a given cell in the matrix."""
    neighbors = []
    for i in range(max(0, row-1), min(row+2, matrix.shape[0])):
        for j in range(max(0, col-1), min(col+2, matrix.shape[1])):
            if (i, j) != (row, col):
                neighbors.append(matrix[i, j])
    return neighbors

def check_connectivity(matrix):
    """Check the connectivity of the matrix using OpenCV."""
    num_labels, labels = cv.connectedComponents(matrix, connectivity=4)
    return num_labels == 2  # 2 because one label is the background

# Example usage
matrix = cv.imread('img_data/drawn_image.png', cv.IMREAD_GRAYSCALE)  # Load the image as a grayscale matrix
cv.imshow('Original Image', matrix)
cv.waitKey(0)
cv.destroyAllWindows()
print("Matrix:")
print(matrix)

row, col = 2, 2
neighbors = get_neighbors(matrix, row, col)
print(f"Neighbors of cell ({row}, {col}): {neighbors}")

is_connected = check_connectivity(matrix)
print(f"Matrix is {'connected' if is_connected else 'not connected'}")

# Display the matrix using OpenCV
cv.imshow('Matrix', matrix * 255)  # Multiply by 255 to visualize binary matrix
cv.waitKey(0)
cv.destroyAllWindows()