import cv2
import numpy as np
import lip
import matplotlib.pyplot as plt

# Step 1: Load the image in grayscale
img = cv2.imread('img_data/lena.jpg', cv2.IMREAD_GRAYSCALE)

#img = lip.AddPeriodicNoise(img, frequency=20, amplitude=100.5, angle=45)

cv2.imshow("Image with Periodic Noise", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Validate image loading
if img is None:
    raise ValueError("Failed to load the image.")

# Step 2: Compute the 2D DFT (complex)
dft = lip.FourierTransform2D(img)

# Step 3: Separate magnitude and phase
magnitude = np.abs(dft)
phase = np.angle(dft)

# Step 4: Direct visualization of the magnitude (no log)
plt.figure(figsize=(15, 4))
plt.subplot(1, 4, 1)
plt.title("Direct Magnitude")
plt.imshow(magnitude, cmap='gray')
plt.axis('off')

# Step 5: Apply logarithm to the magnitude
log_magnitude = np.log1p(magnitude)

plt.subplot(1, 4, 2)
plt.title("Log(Magnitude)")
plt.imshow(log_magnitude, cmap='gray')
plt.axis('off')

# Step 6: Center the spectrum (move DC to the center)
log_magnitude_shifted = np.fft.fftshift(log_magnitude)

plt.subplot(1, 4, 3)
plt.title("Centered Log(Magnitude)")
plt.imshow(log_magnitude_shifted, cmap='gray')
plt.axis('off')

# Step 7: Visualize the raw phase (unnormalized)
plt.subplot(1, 4, 4)
plt.title("Unnormalized Phase")
plt.imshow(phase, cmap='gray')
plt.axis('off')

plt.suptitle("Step-by-Step Visualization: Magnitude and Phase", fontsize=16)
plt.tight_layout()
plt.show()

# Step 8: Normalize the phase for visualization
phase_shifted = np.fft.fftshift(phase)
phase_normalized = cv2.normalize(phase_shifted, None, 0, 255, cv2.NORM_MINMAX)
phase_normalized = phase_normalized.astype(np.uint8)

# Step 9: Display all final results together
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Centered Log-Magnitude Spectrum")
plt.imshow(log_magnitude_shifted, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Normalized Phase")
plt.imshow(phase_normalized, cmap='gray')
plt.axis('off')

plt.suptitle("2D DFT: Final Results", fontsize=16)
plt.tight_layout()
plt.show()
