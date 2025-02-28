import cv2
import numpy as np

def segment_image_with_connected_components(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Binarize the image
    #_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(gray, connectivity=8)

    return labels, num_labels

# Example usage
image = cv2.imread('img_data/arte.jpg')  # Replace with your image path
labels, num_labels = segment_image_with_connected_components(image)

print(f"Number of connected components: {num_labels}")

# Display the segmented image with labels
cv2.imshow('Segmented Image', labels.astype(np.uint8) * 10)  # Multiply by 10 for better visibility
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Number of connected components: {num_labels}")
