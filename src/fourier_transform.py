import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def fft1d(signal):
    N = len(signal)
    if N <= 1:
        return signal
    if N % 2 != 0:
        raise ValueError("Signal length must be a power of 2")

    even = fft1d(signal[::2])
    odd = fft1d(signal[1::2])

    factor = np.exp(-2j * np.pi * np.arange(N) / N)
    return np.concatenate([even + factor[:N // 2] * odd,
                           even - factor[:N // 2] * odd])

def fft2d(image):
    M, N = image.shape
    if (M & (M - 1)) != 0 or (N & (N - 1)) != 0:
        raise ValueError("Image dimensions must be powers of 2")

    row_fft = np.array([fft1d(row) for row in image])
    col_fft = np.array([fft1d(col) for col in row_fft.T]).T

    return col_fft

def ifft1d(signal):
    N = len(signal)
    if N <= 1:
        return signal
    if N % 2 != 0:
        raise ValueError("Signal length must be a power of 2")

    even = ifft1d(signal[::2])
    odd = ifft1d(signal[1::2])

    factor = np.exp(2j * np.pi * np.arange(N) / N)
    return np.concatenate([even + factor[:N // 2] * odd,
                           even - factor[:N // 2] * odd]) / 2

def ifft2d(freq_image):
    col_ifft = np.array([ifft1d(col) for col in freq_image.T]).T
    row_ifft = np.array([ifft1d(row) for row in col_ifft])
    return row_ifft

# --- Example usage ---
if __name__ == "__main__":
    image = cv.imread('img_data/lena.jpg', cv.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not load the image.")
        exit()

    image = cv.resize(image, (256, 256))

    F = fft2d(image)
    F_magnitude = np.log(np.abs(F) + 1)

    reconstructed = ifft2d(F)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title("Original Image")
    axs[1].imshow(np.fft.fftshift(F_magnitude), cmap='gray')
    axs[1].set_title("Magnitude Spectrum (log)")
    axs[2].imshow(np.abs(reconstructed), cmap='gray')  # Use np.abs to handle complex values
    axs[2].set_title("Reconstructed Image")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()