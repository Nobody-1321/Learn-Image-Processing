import cv2 as cv
import matplotlib.pyplot as plt
import lip

# Load the image

path = lip.parse_args_path()

img = cv.imread(path)
if img is None:
    print("Error: Could not load the image.")
    exit()

# Split the color channels
b, g, r = lip.channels_bgr(img)

# Prepare data for the channels
channels = [
    {"name": "Blue", "image": b, "color": "blue"},
    {"name": "Green", "image": g, "color": "green"},
    {"name": "Red", "image": r, "color": "red"},
]

# Initialize the index for the displayed channel
current_channel_idx = 0

# Function to display a channel and its histogram
def display_channel(idx):
    
    channel = channels[idx]
    image = channel["image"]
    color = channel["color"]

    # Compute histogram
    if idx == 0:
        hist = cv.calcHist([image], [0], None, [256], [0, 256])
    if idx == 1:
        hist = cv.calcHist([image], [1], None, [256], [0, 256])
    if idx == 2:
        hist = cv.calcHist([image], [2], None, [256], [0, 256])


    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # Clear the current figure
    plt.clf()

    # Display the image
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.title(f"{channel['name']} Channel Image")

    # Display the histogram
    plt.subplot(1, 2, 2)
    plt.plot(hist, color=color)
    plt.title(f"{channel['name']} Channel Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("# of Pixels")
    plt.xlim([0, 256])

    # Redraw the figure
    plt.tight_layout()
    plt.draw()

# Function to handle key events
def on_key(event):
    global current_channel_idx

    if event.key == "right":  # Move to the next channel
        current_channel_idx = (current_channel_idx + 1) % len(channels)
        display_channel(current_channel_idx)
    elif event.key == "left":  # Move to the previous channel
        current_channel_idx = (current_channel_idx - 1) % len(channels)
        display_channel(current_channel_idx)

# Initialize the plot
plt.figure(figsize=(10, 5))
display_channel(current_channel_idx)

# Connect the key press event to the handler
plt.gcf().canvas.mpl_connect("key_press_event", on_key)

# Show the plot
plt.show()

