import lip  # Custom library/module that contains connected components functions
import cv2 as cv
import numpy as np

def apply_color_map(label_image):
    # Assign a random color to each label (component)
    num_labels = np.max(label_image) + 1
    colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background is black

    # Replace each label in the label image with its corresponding color
    color_image = colors[label_image]
    
    return color_image

def overlay_color_map(original_image, color_map):
    # Convert grayscale image to BGR if necessary
    if len(original_image.shape) == 2 or original_image.shape[2] == 1:
        original_image = cv.cvtColor(original_image, cv.COLOR_GRAY2BGR)
    
    # Blend the original image with the color map (50% of each)
    overlay_image = cv.addWeighted(original_image, 0.5, color_map, 0.5, 0)
    return overlay_image

if __name__ == "__main__":

    # Load grayscale image
    # You can choose among several examples
    img = cv.imread("img_data/caballo.webp", cv.IMREAD_GRAYSCALE)
    # img = cv.imread("img_data/astro.jpg", cv.IMREAD_GRAYSCALE)
    # img = cv.imread("img_data/lena.jpg", cv.IMREAD_GRAYSCALE)
    #img = cv.imread("img_data/Rose.jpg", cv.IMREAD_GRAYSCALE)
    # img = cv.imread("img_data/pieces.jpg", cv.IMREAD_GRAYSCALE)
    # img = cv.imread("img_data/arte.jpg", cv.IMREAD_GRAYSCALE)

    # Resize image to standard size
    img = cv.resize(img, (720, 480), interpolation=cv.INTER_NEAREST)

    # Binarize the image (thresholding)
    _, img = cv.threshold(img, 50, 255, cv.THRESH_BINARY)

    # Apply Connected Components using Union-Find
    labeled_img_4 = lip.ConnectedComponentsByUnionFind(img)  # 4-connected components
    labeled_img_8 = lip.connected_components_by_union_find_8_connected(img)  # 8-connected components

    # Normalize label images for grayscale display (0-255)
    normalized_img_4 = cv.normalize(labeled_img_4, None, 0, 255, cv.NORM_MINMAX)
    normalized_img_8 = cv.normalize(labeled_img_8, None, 0, 255, cv.NORM_MINMAX)

    # Convert to 8-bit for display
    normalized_img_4 = np.uint8(normalized_img_4)
    normalized_img_8 = np.uint8(normalized_img_8)

    # Print the number of connected components found
    print("Number of 4-connected components:", np.max(labeled_img_4))
    print("Number of 8-connected components:", np.max(labeled_img_8))

    # Apply random colors to the labeled images
    color_img_4 = apply_color_map(labeled_img_4)
    color_img_8 = apply_color_map(labeled_img_8)

    # Overlay color maps onto original image
    overlay_img_4 = overlay_color_map(img, color_img_4)
    overlay_img_8 = overlay_color_map(img, color_img_8)

    # Apply Gaussian blur to smooth the image
    overlay_img_4 = cv.GaussianBlur(overlay_img_4, (5, 5), 0)
    overlay_img_8 = cv.GaussianBlur(overlay_img_8, (5, 5), 0)

    # Morphological closing to remove small noise
    kernel = np.ones((3,3), np.uint8)
    overlay_img_4 = cv.morphologyEx(overlay_img_4, cv.MORPH_CLOSE, kernel)
    overlay_img_8 = cv.morphologyEx(overlay_img_8, cv.MORPH_CLOSE, kernel)

    # Apply erosion to refine shapes and edges
    overlay_img_4 = cv.erode(overlay_img_4, kernel, iterations=1)
    overlay_img_8 = cv.erode(overlay_img_8, kernel, iterations=1)

    # Display the results in separate windows
    cv.imshow('4-connected', overlay_img_4)
    cv.imshow('8-connected', overlay_img_8)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Show both results together using the custom utility function
    lip.show_images_together([overlay_img_8, overlay_img_4], ["8-connected", "4-connected"])
