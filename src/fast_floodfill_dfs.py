import numpy as np
import cv2 as cv
import lip

# Example usage
if __name__ == "__main__":
    # Create a sample image
    img =  cv.imread('img_data/drawn_image_3.png', cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, (200, 200), interpolation=cv.INTER_NEAREST)

    # Apply flood fill
    seed_point = (100, 100)  # Seed point inside the black square
    new_color = 20  # New color to fill with
    lip.fast_floodfill_dfs(img, seed_point, new_color)

    # Display the final image
    cv.imshow('Flood Fill', img)
    cv.waitKey(0)
    cv.destroyAllWindows()