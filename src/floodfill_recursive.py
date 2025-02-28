import numpy as np
import cv2 as cv

def floodfill_recursive(I, p, new_color):
    orig_color = I[p[1], p[0]]
    if orig_color == new_color:
        return

    def floodfill(I, x, y, orig_color, new_color):
        if x < 0 or x >= I.shape[1] or y < 0 or y >= I.shape[0]:
            return
        if I[y, x] != orig_color:
            return

        I[y, x] = new_color

        # Recursively call floodfill on all 4-connected neighbors
        floodfill(I, x + 1, y, orig_color, new_color)
        floodfill(I, x - 1, y, orig_color, new_color)
        floodfill(I, x, y + 1, orig_color, new_color)
        floodfill(I, x, y - 1, orig_color, new_color)

    floodfill(I, p[0], p[1], orig_color, new_color)

# Example usage
if __name__ == "__main__":
    # Create a sample image
    #img = cv.imread('img_data/drawn_image.png', cv.IMREAD_GRAYSCALE)
    #img = cv.resize(img, (50, 50), interpolation=cv.INTER_NEAREST)
        # Create a sample image
    img = np.ones((10, 10), dtype=np.uint8) * 255
    img[3:7, 3:7] = 0  # Create a black square in the middle

    cv.imshow('Original Image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    print("Original Image:")
    print(img)

    # Apply flood fill
    seed_point = (5, 5)  # Seed point inside the black square
    new_color = 120  # New color to fill with
    floodfill_recursive(img, seed_point, new_color)

    print("Image after Flood Fill:")
    print(img)

    # Display the image using OpenCV
    cv.imshow('Flood Fill', img)
    cv.waitKey(0)
    cv.destroyAllWindows()