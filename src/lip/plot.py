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

def plot_images_and_histograms(img1, img2, img_equalized_manual1, img_equalized_manual2):
    """
    Plots the original images, their histograms, the equalized images, and their histograms.
    
    Parameters:
        img1 (numpy.ndarray): First original grayscale image.
        img2 (numpy.ndarray): Second original grayscale image.
        img_equalized_manual1 (numpy.ndarray): Manually equalized first image.
        img_equalized_manual2 (numpy.ndarray): Manually equalized second image.
    """
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 8))

    # Primera imagen y su histograma
    plt.subplot(2, 2, 1)
    plt.title("Imagen 1")
    plt.imshow(img1, cmap='gray')
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.title("Histograma Imagen 1")
    plt.hist(img1.flatten(), bins=256, range=[0, 256], color='black')

    # Segunda imagen y su histograma
    plt.subplot(2, 2, 3)
    plt.title("Imagen 2")
    plt.imshow(img2, cmap='gray')
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.title("Histograma Imagen 2")
    plt.hist(img2.flatten(), bins=256, range=[0, 256], color='black')

    plt.tight_layout()
    plt.show()

def plot_three_images_and_histograms(img1, img2, img3, title1="Imagen 1", title2="Imagen 2", title3="Imagen 3"):
    """
    Plots three images and their histograms side by side using Matplotlib.
    
    Parameters:
        img1 (numpy.ndarray): First image (grayscale or color).
        img2 (numpy.ndarray): Second image (grayscale or color).
        img3 (numpy.ndarray): Third image (grayscale or color).
        title1 (str): Title for the first image.
        title2 (str): Title for the second image.
        title3 (str): Title for the third image.
    """
    
    plt.figure(figsize=(15, 10))

    # Primera imagen y su histograma
    plt.subplot(3, 2, 1)
    plt.title(title1)
    if len(img1.shape) == 2:  # Imagen en escala de grises
        plt.imshow(img1, cmap='gray')
    else:  # Imagen en color (Convertir de BGR a RGB para matplotlib)
        plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(3, 2, 2)
    plt.title(f"Histograma {title1}")
    if len(img1.shape) == 2:  # Imagen en escala de grises
        plt.hist(img1.flatten(), bins=256, range=[0, 256], color='black')
    else:  # Imagen en color
        colors = ('blue', 'green', 'red')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([img1], [i], None, [256], [0, 256])
            plt.plot(hist, color=color, label=f"{color.capitalize()} Channel")
        plt.legend()

    # Segunda imagen y su histograma
    plt.subplot(3, 2, 3)
    plt.title(title2)
    if len(img2.shape) == 2:  # Imagen en escala de grises
        plt.imshow(img2, cmap='gray')
    else:  # Imagen en color (Convertir de BGR a RGB para matplotlib)
        plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(3, 2, 4)
    plt.title(f"Histograma {title2}")
    if len(img2.shape) == 2:  # Imagen en escala de grises
        plt.hist(img2.flatten(), bins=256, range=[0, 256], color='black')
    else:  # Imagen en color
        colors = ('blue', 'green', 'red')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([img2], [i], None, [256], [0, 256])
            plt.plot(hist, color=color, label=f"{color.capitalize()} Channel")
        plt.legend()

    # Tercera imagen y su histograma
    plt.subplot(3, 2, 5)
    plt.title(title3)
    if len(img3.shape) == 2:  # Imagen en escala de grises
        plt.imshow(img3, cmap='gray')
    else:  # Imagen en color (Convertir de BGR a RGB para matplotlib)
        plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(3, 2, 6)
    plt.title(f"Histograma {title3}")
    if len(img3.shape) == 2:  # Imagen en escala de grises
        plt.hist(img3.flatten(), bins=256, range=[0, 256], color='black')
    else:  # Imagen en color
        colors = ('blue', 'green', 'red')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([img3], [i], None, [256], [0, 256])
            plt.plot(hist, color=color, label=f"{color.capitalize()} Channel")
        plt.legend()

    plt.tight_layout()
    plt.show()

def plot_two_images_and_histograms(img1, img2, title1="Imagen 1", title2="Imagen 2"):
    """
    Plots two images and their histograms side by side using Matplotlib.
    
    Parameters:
        img1 (numpy.ndarray): First image (grayscale or color).
        img2 (numpy.ndarray): Second image (grayscale or color).
        title1 (str): Title for the first image.
        title2 (str): Title for the second image.
    """
    
    plt.figure(figsize=(12, 8))

    # Primera imagen y su histograma
    plt.subplot(2, 2, 1)
    plt.title(title1)
    if len(img1.shape) == 2:  # Imagen en escala de grises
        plt.imshow(img1, cmap='gray')
    else:  # Imagen en color (Convertir de BGR a RGB para matplotlib)
        plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.title(f"Histograma {title1}")
    if len(img1.shape) == 2:  # Imagen en escala de grises
        plt.hist(img1.flatten(), bins=256, range=[0, 256], color='black')
    else:  # Imagen en color
        colors = ('blue', 'green', 'red')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([img1], [i], None, [256], [0, 256])
            plt.plot(hist, color=color, label=f"{color.capitalize()} Channel")
        plt.legend()

    # Segunda imagen y su histograma
    plt.subplot(2, 2, 3)
    plt.title(title2)
    if len(img2.shape) == 2:  # Imagen en escala de grises
        plt.imshow(img2, cmap='gray')
    else:  # Imagen en color (Convertir de BGR a RGB para matplotlib)
        plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.title(f"Histograma {title2}")
    if len(img2.shape) == 2:  # Imagen en escala de grises
        plt.hist(img2.flatten(), bins=256, range=[0, 256], color='black')
    else:  # Imagen en color
        colors = ('blue', 'green', 'red')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([img2], [i], None, [256], [0, 256])
            plt.plot(hist, color=color, label=f"{color.capitalize()} Channel")
        plt.legend()

    plt.tight_layout()
    plt.show()