import numpy as np
import cv2 as cv

def fast_floodfill(I, p, new_color):
    orig_color = I[p[1], p[0]]
    if orig_color == new_color:
        return

    frontier = [p]
    I[p[1], p[0]] = new_color

    while frontier:
        x, y = frontier.pop()
        for q in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
            qx, qy = q
            if 0 <= qx < I.shape[1] and 0 <= qy < I.shape[0] and I[qy, qx] == orig_color:
                frontier.append(q)
                I[qy, qx] = new_color

                # Visualize the process step by step
                cv.imshow('Flood Fill', I)
                cv.waitKey(1)

# Example usage
if __name__ == "__main__":
    # Create a sample image
    img =  cv.imread('img_data/drawn_image_3.png', cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, (100, 100), interpolation=cv.INTER_NEAREST)

    print("Original Image:")
    print(img)

    # Apply flood fill
    seed_point = (50, 30)  # Seed point inside the black square
    new_color = 128  # New color to fill with
    fast_floodfill(img, seed_point, new_color)

    print("Image after Flood Fill:")
    print(img)

    # Display the final image
    cv.imshow('Flood Fill', img)
    cv.waitKey(0)
    cv.destroyAllWindows()