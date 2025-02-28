import numpy as np
import cv2 as cv

def floodfill_iterative(I, p, new_color):
    orig_color = I[p[1], p[0]]
    if orig_color == new_color:
        return

    stack = [p]
    while stack:
        x, y = stack.pop()
        if x < 0 or x >= I.shape[1] or y < 0 or y >= I.shape[0]:
            continue
        if I[y, x] != orig_color:
            continue

        I[y, x] = new_color

        # Add all 4-connected neighbors to the stack
        stack.append((x + 1, y))
        stack.append((x - 1, y))
        stack.append((x, y + 1))
        stack.append((x, y - 1))

        # Visualize the process step by step
        cv.imshow('Flood Fill', I)
        cv.waitKey(1)  # Pause for 100 milliseconds

# Example usage
if __name__ == "__main__":
    # Create a sample image
    img = cv.imread('img_data/drawn_image.png', cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, (100, 100), interpolation=cv.INTER_NEAREST)

    print("Original Image:")
    print(img)

    # Apply flood fill
    seed_point = (5, 5)  # Seed point inside the black square
    new_color = 128  # New color to fill with
    floodfill_iterative(img, seed_point, new_color)

    print("Image after Flood Fill:")
    print(img)

    # Display the final image
    cv.imshow('Flood Fill', img)
    cv.waitKey(0)
    cv.destroyAllWindows()