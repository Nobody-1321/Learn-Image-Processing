import cv2 as cv
import matplotlib.pyplot as plt
import ena

# Load the image
path = ena.parse_args_path()

img = cv.imread(path)

if img is None:
    print("Error: Could not load the image.")
    exit()

# Split the color channels
b, g, r = cv.split(img)

# Compute the histograms
hist_b = cv.calcHist([b], [0], None, [256], [0, 256])
hist_g = cv.calcHist([g], [0], None, [256], [0, 256])
hist_r = cv.calcHist([r], [0], None, [256], [0, 256])

# Configure subplots
plt.figure(figsize=(12, 6))

# Display the original image
plt.subplot(1, 2, 1)
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Convert from BGR to RGB
plt.imshow(img_rgb)
plt.axis("off")
plt.title("Image")

# Display the histograms
plt.subplot(1, 2, 2)
plt.title("Color Channel Histograms")
plt.xlabel("Intensity")
plt.ylabel("# of Pixels")
plt.plot(hist_b, color="blue", label="Blue Channel")
plt.plot(hist_g, color="green", label="Green Channel")
plt.plot(hist_r, color="red", label="Red Channel")
plt.xlim([0, 256])
plt.legend()

# Show the figure
plt.tight_layout()
plt.show()
