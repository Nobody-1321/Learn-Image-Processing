import numpy as np
import cv2 as cv

drawing = False  # True if the mouse is pressed

# Callback function to handle mouse events
def draw(event, x, y, flags, param):
    global drawing
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        cv.circle(img, (x, y), 5, (0, 0, 0), -1)  # Draw a black circle
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            cv.circle(img, (x, y), 5, (0, 0, 0), -1)  # Draw a black circle
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False

# Create a white image
img = np.ones((500, 500, 3), dtype=np.uint8) * 255

# Create a window and set the mouse callback function
cv.namedWindow('Draw')
cv.setMouseCallback('Draw', draw)

while True:
    cv.imshow('Draw', img)
    key = cv.waitKey(1) & 0xFF
    if key == 27:  # Press 'Esc' to exit
        break
    elif key == ord('s'):  # Press 's' to save the image
        cv.imwrite('drawn_image.png', img)
        print("Image saved as 'drawn_image.png'")

cv.destroyAllWindows()