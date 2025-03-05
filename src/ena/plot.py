import cv2 
import numpy 
import matplotlib.pyplot as plt

import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_histogram(img):
    """
    Display the histogram of a grayscale or color image.
    
    Parameters:
    - image (numpy.ndarray): A 2D (grayscale) or 3D (color) array representing the image.
    
    Returns:
    - None
    """

    # Verificar si la imagen es en color o en escala de grises
    if len(img.shape) == 2:  # Imagen en escala de grises
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        plt.plot(hist, color='black', label="Grayscale")
    else:  # Imagen en color
        colors = ('blue', 'green', 'red')
        channels = ('Blue', 'Green', 'Red')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(hist, color=color, label=f"{channels[i]} Channel")

    # Configurar la gráfica
    plt.title("Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Mostrar histograma
    plt.show()

def compare_histograms(img1, img2, title1="Image 1", title2="Image 2"):
    """
    Compute and display the histograms of two images in the same plot.
    
    Parameters:
    - img1 (numpy.ndarray): First image (grayscale or color).
    - img2 (numpy.ndarray): Second image (grayscale or color).
    - title1 (str): Label for the first image histogram.
    - title2 (str): Label for the second image histogram.
    
    Returns:
    - None
    """

    plt.figure(figsize=(10, 5))  # Crear una figura más ancha

    # Si ambas imágenes son en escala de grises
    if len(img1.shape) == 2 and len(img2.shape) == 2:
        hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])

        plt.plot(hist1, color='black', linestyle='-', label=f"{title1} (Grayscale)")
        plt.plot(hist2, color='gray', linestyle='--', label=f"{title2} (Grayscale)")

    else:  # Si las imágenes son en color
        colors = ('blue', 'green', 'red')
        channels = ('Blue', 'Green', 'Red')

        for i, color in enumerate(colors):
            hist1 = cv2.calcHist([img1], [i], None, [256], [0, 256])
            hist2 = cv2.calcHist([img2], [i], None, [256], [0, 256])

            plt.plot(hist1, color=color, linestyle='-', label=f"{title1} {channels[i]} Channel")
            plt.plot(hist2, color=color, linestyle='--', label=f"{title2} {channels[i]} Channel")

    # Configuración del gráfico
    plt.title("Histogram Comparison")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Mostrar los histogramas
    plt.show()

def show_image_with_histogram(img, title="Image"):
    """
    Display an image alongside its histogram.
    
    Parameters:
    - image (numpy.ndarray): The input image (grayscale or color).
    - title (str): Title for the image.
    
    Returns:
    - None
    """

    # Crear una figura con dos subgráficos
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Mostrar la imagen en la primera subgráfica
    if len(img.shape) == 2:  # Imagen en escala de grises
        axes[0].imshow(img, cmap='gray')
    else:  # Imagen en color (Convertir de BGR a RGB para matplotlib)
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    axes[0].set_title(title)
    axes[0].axis("off")

    # Calcular y mostrar el histograma en la segunda subgráfica
    axes[1].set_title("Histogram")
    axes[1].set_xlabel("Pixel Intensity")
    axes[1].set_ylabel("Frequency")

    if len(img.shape) == 2:  # Imagen en escala de grises
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        axes[1].plot(hist, color='black')
    else:  # Imagen en color
        colors = ('blue', 'green', 'red')
        channels = ('Blue', 'Green', 'Red')
        
        for i, color in enumerate(colors):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            axes[1].plot(hist, color=color, label=channels[i])

        axes[1].legend()

    axes[1].grid(True, linestyle='--', alpha=0.5)

    # Mostrar la figura con la imagen y su histograma
    plt.tight_layout()
    plt.show()

