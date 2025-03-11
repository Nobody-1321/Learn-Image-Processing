import cv2 as cv
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import factorial

def channels_bgr(img):
    """
    Splits the input BGR image into its individual blue, green, and red channels.

    Parameters:
        img (numpy.ndarray): Input image in BGR format.

    Returns:
        tuple: A tuple containing three images:
            - blue_img (numpy.ndarray): Image with only the blue channel.
            - green_img (numpy.ndarray): Image with only the green channel.
            - red_img (numpy.ndarray): Image with only the red channel.
    """

    b, g, r = cv.split(img)
    zeros = np.zeros(img.shape[:2], dtype="uint8")

    blue_img = cv.merge([b, zeros, zeros])
    green_img = cv.merge([zeros, g, zeros])
    red_img = cv.merge([zeros, zeros, r])

    return blue_img, green_img, red_img

def channels_hsv(img):
        """
        Splits an image into its HSV (Hue, Saturation, Value) channels.
        Parameters:
            img (numpy.ndarray): Input image in BGR format.
        Returns:
            tuple: A tuple containing three images:
                - hue_img (numpy.ndarray): Image with only the hue channel.
                - saturation_img (numpy.ndarray): Image with only the saturation channel.
                - value_img (numpy.ndarray): Image with only the value channel.
        """
        
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        zeros = np.zeros(img.shape[:2], dtype="uint8")
        
        hue_img = cv.merge([h, zeros, zeros])
        saturation_img = cv.merge([zeros, s, zeros])
        value_img = cv.merge([zeros, zeros, v])
        
        return hue_img, saturation_img, value_img

def open_image(path):
    """
    Opens an image from the specified file path.

    Parameters:
    path (str): The file path to the image.

    Returns:
    img: The image read from the file path.
    """
    img = cv.imread(path)
    return img
    
def resize_image(img, width, height):
    """
    Resize the given image to the specified width and height.

    Args:
        img (numpy.ndarray): The input image to be resized.
        width (int): The desired width of the resized image.
        height (int): The desired height of the resized image.

    Returns:
        numpy.ndarray: The resized image.
    """
    return cv.resize(img, (width, height), interpolation=cv.INTER_AREA)

def save(filename, img):
    """
    Save an image to a file.

    Parameters:
        filename (str): The path to the file where the image will be saved.
        img (numpy.ndarray): The image to be saved.

    Raises:
        Exception: If there is an error saving the image, an exception is caught and an error message is printed.
    """

    try:
        cv.imwrite(filename, img)
    except Exception as e:
        print(f"Error saving image: {e}")   

def combine_channels(img, red, green, blue):
    """
    Combines separate red, green, and blue channels into a single image.

    Parameters:
        img (numpy.ndarray): The original image used for reference dimensions.
        red (numpy.ndarray): The red channel image.
        green (numpy.ndarray): The green channel image.
        blue (numpy.ndarray): The blue channel image.
    Returns:
        numpy.ndarray: The combined image with merged channels, or None if the channel dimensions do not match the original image.
    """
    
    if red.shape[:2] != img.shape[:2] or green.shape[:2] != img.shape[:2] or blue.shape[:2] != img.shape[:2]:
        return None
        
    return cv.merge([blue, green, red])

def flip_flop_flipflop(img):
    """
    Applies a series of flip operations to the input image.

    Parameters:
        img (numpy.ndarray): The input image to be flipped.

    Returns:
    tuple: A tuple containing three images:
        - The first image is flipped both horizontally and vertically.
        - The second image is flipped horizontally.
        - The third image is flipped both horizontally and vertically (same as the first image).
    """
    imgc = img.copy()
    return cv.flip(img, -1), cv.flip(imgc, 1), cv.flip(img, -1)

def rotate_image(img, angle):
    rows, cols = img.shape[:2]
    M = cv.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    return cv.warpAffine(img, M, (cols, rows))

def get_combine_channels_rg_rb_gb(img):
    """
    Splits the input image into its blue, green, and red channels, and then combines them into three new images:
    - Red-Green (RG)
    - Red-Blue (RB)
    - Green-Blue (GB)
    Args:
        img (numpy.ndarray): The input image in BGR format.
    Returns:
        tuple: A tuple containing three images:
            - red_green (numpy.ndarray): Image with combined red and green channels.
            - red_blue (numpy.ndarray): Image with combined red and blue channels.
            - green_blue (numpy.ndarray): Image with combined green and blue channels.
    """
    blue, green, red = cv.split(img)
    zeros = np.zeros(img.shape[:2], dtype="uint8")
    
    red_green = cv.merge([zeros, green, red])
    red_blue = cv.merge([blue, zeros, red])
    green_blue = cv.merge([blue, green, zeros])
    
    return red_green, red_blue, green_blue

def HistogramEqualization(img):
    """
    Apply histogram equalization to an input image.
    
    Parameters:
        img (numpy.ndarray): Input grayscale image as a 2D numpy array.
    
    Returns:
        numpy.ndarray: Image after applying histogram equalization, with the same shape as the input image.
    """
    # 1. Calcular el histograma
    hist, bins = np.histogram(img.flatten(), bins=256, range=[0, 256])
    
    # 2. Calcular la función de distribución acumulativa (CDF)
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]  # Normalizar el CDF para que vaya de 0 a 1
    
    # 3. Mapear los valores originales a los nuevos valores usando la CDF
    cdf_scaled = (cdf_normalized * 255).astype(np.uint8)  # Escalar a [0, 255]
    img_equalized = cdf_scaled[img]  # Usar la CDF como lookup table
    
    return img_equalized

def HistogramMatchingGray(img_ref, img_target):
    """
    Perform histogram matching on a grayscale image.

    Parameters:
        img_ref (numpy.ndarray): Reference grayscale image.
        img_target (numpy.ndarray): Input grayscale image to be matched.

    Returns:
        numpy.ndarray: The transformed image with a histogram matching that of the reference image.

    """

    # 1. Compute histogram of the reference image
    hist_ref, bins = np.histogram(img_ref.flatten(), bins=256, range=[0, 256])
    
    # 2. Compute histogram of the input image
    hist, bins = np.histogram(img_target.flatten(), bins=256, range=[0, 256])
    
    # 3. Compute the cumulative distribution function (CDF) for both images
    cdf_ref = hist_ref.cumsum()
    cdf_ref_normalized = cdf_ref / cdf_ref[-1]  # Normalize to range [0,1]

    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]  # Normalize to range [0,1]
    
    # 4. Create lookup table for histogram mapping
    lookup_table = np.interp(cdf_normalized, cdf_ref_normalized, range(256)).astype(np.uint8)
    
    # 5. Apply the transformation to the input image
    img_matched = lookup_table[img_target]
    
    return img_matched

def HistogramMatchingRGB(img_ref, img_target):
    """
    Perform histogram matching on an RGB image.

    Parameters:
        img_ref (numpy.ndarray): Reference RGB image.
        img_target (numpy.ndarray): Target RGB image to be matched.

    Returns:
    numpy.ndarray: The transformed image with a histogram matching that of the reference image.
    """

    def match_histogram(channel_ref, channel_target):
        """Apply histogram matching to a single channel."""
        hist_ref, _ = np.histogram(channel_ref.flatten(), bins=256, range=[0, 256])
        hist_target, _ = np.histogram(channel_target.flatten(), bins=256, range=[0, 256])
        
        cdf_ref = hist_ref.cumsum() / hist_ref.sum()
        cdf_target = hist_target.cumsum() / hist_target.sum()
        
        lookup_table = np.interp(cdf_target, cdf_ref, np.arange(256)).astype(np.uint8)
        return lookup_table[channel_target]

    # Convert images to uint8 format (if not already)
    img_ref = img_ref.astype(np.uint8)
    img_target = img_target.astype(np.uint8)

    # Split the images into R, G, B channels
    matched_channels = [
        match_histogram(img_ref[:, :, i], img_target[:, :, i]) for i in range(3)
    ]

    # Merge the transformed channels back into an RGB image
    img_matched = cv.merge(matched_channels)

    return img_matched

def CalHistogram(img):
    
    """
    Compute the histogram of one-channel grayscale image.

    Parameters:
        img (numpy.ndarray): The input grayscale image.

    Returns:
        numpy.ndarray: The computed histogram with 256 bins.
    """
    # Compute the histogram

    hist, bins = np.histogram(img.flatten(), bins=256, range=[0, 256])
    
    return hist

def BgrToGray(img):
    """
    Convert a BGR image to a grayscale image.
    This function takes an image in BGR format and converts it to a grayscale image.
    
    Parameters:
        - img (numpy.ndarray): Input image in BGR format.
    Returns:
        - numpy.ndarray: Grayscale image.
    """
    
    img = img.astype(np.float32)

    blue, green, red = cv.split(img)
    
    gray_img = 0.299 * red + 0.587 * green + 0.114 * blue
    
    gray_img = np.clip(gray_img, 0, 255)

    return gray_img.astype(np.uint8)
    
def floodfill_separate_output(I, O, p, new_color):
    """
    Perform a flood fill on an output image O without modifying the original image I.

    Parameters:
    I (numpy.ndarray): The original input image.
    O (numpy.ndarray): The output image to be modified.
    p (tuple): The starting point (x, y) for the flood fill.
    new_color (int or tuple): The new color to apply to the filled area.

    Returns:
    None
    """
    """Flood Fill en una imagen de salida O sin modificar la imagen original I"""
    orig_color = I[p[1], p[0]]
    if orig_color == 0:  # Evita llenar el fondo (asumimos fondo en 0)
        return  

    stack = [p]  # Pila para DFS
    O[p[1], p[0]] = new_color  # Asigna la etiqueta

    while stack:
        x, y = stack.pop()

        # Vecinos en 4 direcciones
        for q in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
            qx, qy = q

            # Verificar límites y si el píxel puede ser etiquetado
            if 0 <= qx < I.shape[1] and 0 <= qy < I.shape[0]:
                if I[qy, qx] == orig_color and O[qy, qx] == 0:
                    stack.append((qx, qy))
                    O[qy, qx] = new_color

def connected_components_by_repeated_floodfill(I):
    """
    Finds connected components using repeated Flood Fill.

    Parameters:
        I (numpy.ndarray): Input binary image where object pixels are greater than 0.

    Returns:
        numpy.ndarray: Label image where each connected component has a unique label value.
    """
    L = np.zeros_like(I, dtype=np.int32)  # Label image (initially 0)
    next_label = 1  # First label

    # Traverse each pixel of the image
    for y in range(I.shape[0]):
        for x in range(I.shape[1]):
            if I[y, x] > 0 and L[y, x] == 0:  # If it is an object pixel and has no label
                floodfill_separate_output(I, L, (x, y), next_label)
                next_label += 1  # New label for the next component

    return L  # Labeled image

def show_two_images(img1, img2, title="Comparison", orientation="horizontal"):
    """
    Displays two images side by side in a single window using OpenCV.
    
    Parameters:
    - img1: First image (numpy array).
    - img2: Second image (numpy array).
    - title: Title of the window.
    - orientation: "horizontal" (default) or "vertical".
    """
    # Ensure both images have the same number of channels
    if len(img1.shape) == 2:  # If grayscale, convert to 3 channels
        img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:
        img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)

    # Resize the smaller image to match the size
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    if orientation == "horizontal":
        if h1 != h2:
            max_h = max(h1, h2)
            img1 = cv.resize(img1, (w1, max_h))
            img2 = cv.resize(img2, (w2, max_h))
        combined = np.hstack((img1, img2))  # Stack horizontally
    else:
        if w1 != w2:
            max_w = max(w1, w2)
            img1 = cv.resize(img1, (max_w, h1))
            img2 = cv.resize(img2, (max_w, h2))
        combined = np.vstack((img1, img2))  # Stack vertically

    # Display the combined image
    cv.imshow(title, combined)

class UnionFind:
    """Estructura de datos Union-Find con compresión de caminos."""
    def __init__(self, size):
        self.parent = np.arange(size)
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Compresión de caminos
        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_y] = root_x

def connected_components_by_union_find(image):
    """
    Perform connected components labeling on a binary image using the Union-Find algorithm.
    Parameters:
    image (numpy.ndarray): A 2D binary image where foreground pixels are non-zero and background pixels are zero.
    Returns:
    numpy.ndarray: A 2D array of the same shape as the input image, where each connected component is assigned a unique label.
    """
    height, width = image.shape
    label_image = np.zeros((height, width), dtype=int)
    uf = UnionFind(height * width)  # Estructura para uniones
    next_label = 1

    # Primera pasada: Asignación preliminar y registro de equivalencias
    for y in range(height):
        for x in range(width):
            if image[y, x] == 0:  # Fondo (asumimos 0 como fondo)
                continue
            
            neighbors = []
            if x > 0 and image[y, x] == image[y, x - 1]:  # Vecino izquierdo
                neighbors.append(label_image[y, x - 1])
            if y > 0 and image[y, x] == image[y - 1, x]:  # Vecino superior
                neighbors.append(label_image[y - 1, x])
            
            if neighbors:
                min_label = min(neighbors)
                label_image[y, x] = min_label
                for label in neighbors:
                    uf.union(min_label, label)
            else:
                label_image[y, x] = next_label
                next_label += 1

    # Segunda pasada: Reetiquetar con los representantes equivalentes
    labels_flat = label_image.flatten()
    labels_flat = np.vectorize(uf.find)(labels_flat)  # Vectorizado en NumPy
    label_image = labels_flat.reshape(height, width)

    return label_image

def connected_components_by_union_find_8_connected(image):
    """
    Algoritmo de etiquetado de componentes conexas con Union-Find (8 vecinos), optimizado con NumPy.
    """
    height, width = image.shape
    label_image = np.zeros((height, width), dtype=int)
    uf = UnionFind(height * width)  # Estructura para unir etiquetas
    next_label = 1
    
    # 🔹 Primera pasada: Asignar etiquetas iniciales y registrar equivalencias
    for y in range(height):
        for x in range(width):
            if image[y, x] == 0:  # Fondo
                continue

            # 🔸 Buscar etiquetas de vecinos (usando NumPy para eficiencia)
            neighbors = []
            if x > 0 and image[y, x] == image[y, x - 1]:  # Izquierda
                neighbors.append(label_image[y, x - 1])
            if y > 0 and image[y, x] == image[y - 1, x]:  # Arriba
                neighbors.append(label_image[y - 1, x])
            if x > 0 and y > 0 and image[y, x] == image[y - 1, x - 1]:  # Arriba-Izquierda
                neighbors.append(label_image[y - 1, x - 1])
            if x < width - 1 and y > 0 and image[y, x] == image[y - 1, x + 1]:  # Arriba-Derecha
                neighbors.append(label_image[y - 1, x + 1])
            
            if neighbors:
                min_label = min(neighbors)
                label_image[y, x] = min_label
                for label in neighbors:
                    uf.union(min_label, label)
            else:
                label_image[y, x] = next_label
                next_label += 1

    # 🔹 Segunda pasada: Asignar etiquetas finales con compresión de caminos
    labels_flat = label_image.flatten()
    labels_flat = np.vectorize(uf.find)(labels_flat)  # Vectorizado en NumPy
    label_image = labels_flat.reshape(height, width)

    return label_image

def ConvolveSeparable(I, gh, gv):
    """
    Aplica la convolución separable a la imagen I usando los kernels 1D gh (horizontal) y gv (vertical).

    Parámetros:
    - I: Imagen de entrada (numpy array 2D).
    - gh: Kernel 1D para la convolución horizontal.
    - gv: Kernel 1D para la convolución vertical.

    Retorna:
    - Ir: Imagen después de la convolución con los dos kernels.
    """

    height, width = I.shape
    w = len(gh)
    pad_size = w // 2

    # Paso 1: Padding de la imagen
    I_padded = np.pad(I, ((pad_size, pad_size), (pad_size, pad_size)), mode='reflect')

    # Paso 2: Convolución horizontal
    Itmp = np.zeros_like(I_padded, dtype=np.float32)
    for y in range(height):
        for x in range(width):
            val = 0
            for i in range(w):
                val += gh[i] * I_padded[y + pad_size, x + i]
            Itmp[y + pad_size, x + pad_size] = val
    
    # Paso 3: Convolución vertical
    Ir = np.zeros_like(I_padded, dtype=np.float32)
    for y in range(height):
        for x in range(width):
            val = 0
            for i in range(w):
                val += gv[i] * Itmp[y + i, x + pad_size]
            Ir[y + pad_size, x + pad_size] = val
    
            
    # Eliminar el padding
    #Ir = Ir[pad_size:-pad_size, pad_size:-pad_size]
    Itmp = Itmp[pad_size:-pad_size, pad_size:-pad_size]



    return Itmp

def GetKernelHalfWidth(sigma):
    """
    Computes the half-width of the Gaussian kernel based on the given standard deviation.

    Parameters:
    - sigma (float): The standard deviation of the Gaussian function.

    Returns:
    - half_width (int): The computed half-width of the kernel.
    """
    return int(2.5 * sigma + 0.5)  # Ensures enough coverage of the Gaussian function

def CreateGaussianKernel(sigma):
    """
    Generates a 1D Gaussian kernel based on a given standard deviation (sigma).

    Parameters:
    - sigma (float): The standard deviation of the Gaussian function.

    Returns:
    - gauss (numpy.ndarray): The normalized 1D Gaussian kernel.
    """
    # Step 1: Determine the half-width of the kernel
    half_width = GetKernelHalfWidth(sigma)

    # Step 2: Compute the full width (must be an odd number)
    w = 2 * half_width + 1

    # Step 3: Initialize the kernel and normalization factor
    gauss = np.zeros(w, dtype=np.float32)
    norm = 0.0

    # Step 4: Compute Gaussian values and accumulate normalization factor
    for i in range(w):
        x = i - half_width  # Shift the index to center around 0
        gauss[i] = np.exp(- (x**2) / (2 * sigma**2))
        norm += gauss[i]

    # Step 5: Normalize the kernel so that its sum equals 1
    gauss /= norm

    return gauss

def ConvolveBox(f, w):
    """
    Convolve a 1D signal f with a 1D box kernel of length w.

    Parameters:
    - f: 1D signal (numpy array) with length n.
    - w: Length of the 1D box kernel (odd number).

    Returns:
    - result: The convolution of the signal f with the box kernel.
    """
    n = len(f)
    half_w = w // 2  # Half width of the kernel
    
    # Initialize the result array
    result = np.zeros_like(f, dtype=np.float32)
    
    # Convolution process
    for i in range(n):
        # Sum the values in the window centered at i (with kernel size w)
        val = 0
        for j in range(-half_w, half_w + 1):
            if 0 <= i + j < n:
                val += f[i + j]
        
        # Store the result (normalized average of the window)
        result[i] = val / w  # Normalize by dividing by the kernel size (w)
    
    return result

def ConvolveBox1(f, w):
    """
    Apply a 1D box kernel to the image f (convolution with a box filter).
    
    Parameters:
    - f: The input image (numpy array).
    - w: The width of the box filter (odd integer).
    
    Returns:
    - The convolved image.
    """
    # Ensure the kernel is normalized
    kernel = np.ones(w) / w  # Box filter (normalized)
    
    # Pad the image to handle edges
    pad_size = w // 2
    f_padded = np.pad(f, ((pad_size, pad_size), (pad_size, pad_size)), mode='edge')

    # Output image
    result = np.zeros_like(f, dtype=np.float32)

    # Apply the convolution horizontally
    for y in range(f.shape[0]):
        for x in range(f.shape[1]):
            result[y, x] = np.sum(f_padded[y:y+w, x] * kernel)
    
    return result

def CreateGaussianDerivativeKernel(sigma):
    """
    Generates a 1D Gaussian derivative kernel for edge detection.

    Parameters:
    - sigma (float): Standard deviation of the Gaussian function.

    Returns:
    - gauss_deriv (numpy.ndarray): The 1D Gaussian derivative kernel.
    """
    # Paso 1: Obtener la mitad del ancho del kernel
    half_width = GetKernelHalfWidth(sigma)
    
    # Paso 2: Calcular el ancho total (siempre impar)
    w = 2 * half_width + 1

    # Paso 3: Inicializar el kernel y el factor de normalización
    gauss_deriv = np.zeros(w, dtype=np.float32)
    norm = 0.0

    # Paso 4: Calcular la derivada de la función Gaussiana
    for i in range(w):
        x = i - half_width  # Centramos los valores alrededor de 0
        gauss_deriv[i] = -x * np.exp(- (x ** 2) / (2 * sigma ** 2))  # Derivada de Gauss
        norm += abs(x * gauss_deriv[i])  # Normalización basada en la suma de valores absolutos

    # Paso 5: Normalizar el kernel
    gauss_deriv /= norm

    return gauss_deriv

def SobelOperator(image):
    """
    Aplica el operador de Sobel a una imagen en escala de grises.
    
    Parámetros:
    - image: Imagen de entrada (numpy array 2D en escala de grises).

    Retorna:
    - Gx: Gradiente en la dirección X.
    - Gy: Gradiente en la dirección Y.
    - G: Magnitud del gradiente combinando Gx y Gy.
    """

    #convertir a 32 bits y normalizar
    
    image = image.astype(np.float32)

    # Definir los kernels de Sobel
    Sobel_x = np.array([[-1, 0, 1], 
                         [-2, 0, 2], 
                         [-1, 0, 1]], dtype=np.float32)

    Sobel_y = np.array([[-1, -2, -1], 
                         [0,  0,  0], 
                         [1,  2,  1]], dtype=np.float32)

    Sobel_x = Sobel_x * 1/8
    Sobel_y = Sobel_y * 1/8

    # Aplicar la convolución con los filtros de Sobel
    Gx = cv.filter2D(image, cv.CV_32F, Sobel_x)
    Gy = cv.filter2D(image, cv.CV_32F, Sobel_y)

    # Calcular la magnitud del gradiente
    G = np.sqrt(Gx**2 + Gy**2)

    # Normalizar para visualización
    Gx = cv.normalize(Gx, None, 0, 255, cv.NORM_MINMAX)
    Gy = cv.normalize(Gy, None, 0, 255, cv.NORM_MINMAX)
    G = cv.normalize(G, None, 0, 255, cv.NORM_MINMAX)

    return Gx.astype(np.uint8), Gy.astype(np.uint8), G.astype(np.uint8)

def ScharOperator(image):
        # Definir los kernels de Sobel
    Schar_x = np.array([[-3, 0, 3], 
                         [-10, 0, 10], 
                         [-3, 0, 3]], dtype=np.float32)

    Schar_y = np.array([[ 3, 10,  3], 
                         [0,  0,  0], 
                         [-3,  -10,  -3]], dtype=np.float32)

    Schar_x = Schar_x * 1/32
    Schar_y = Schar_y * 1/32

    # Aplicar la convolución con los filtros de Sobel

    Gx = cv.filter2D(image, cv.CV_32F, Schar_x)
    Gy = cv.filter2D(image, cv.CV_32F, Schar_y)

    # Calcular la magnitud del gradiente
    G = np.sqrt(Gx**2 + Gy**2)

    # Normalizar para visualización
    Gx = cv.normalize(Gx, None, 0, 255, cv.NORM_MINMAX)
    Gy = cv.normalize(Gy, None, 0, 255, cv.NORM_MINMAX)
    G = cv.normalize(G, None, 0, 255, cv.NORM_MINMAX)

    return Gx, Gy, G

def ComputeImageGradient(img, sigma):
    """
    Compute the image gradient using the Gaussian derivative kernel.

    Parameters:
    - img: The input image (numpy array).
    - sigma: The standard deviation of the Gaussian kernel.

    Returns:
    - Gx: The gradient image in the X direction.
    - Gy: The gradient image in the Y direction.
    - G: The magnitude of the gradient.
    """
    # convertir a 32 bits y normalizar
    
    img = img.astype(np.float32)

    # Step 1: Create the Gaussian derivative kernel
    gauss_deriv = CreateGaussianDerivativeKernel(sigma)

    # Step 2: Apply the convolution
    Gx = ConvolveSeparable(img, gauss_deriv, gauss_deriv)
    Gy = ConvolveSeparable(img, gauss_deriv, gauss_deriv)

    # Step 3: Compute the magnitude and phase of the gradient

    # Step 3: Compute the magnitude and phase of the gradient
    Gmag = np.sqrt(Gx**2 + Gy**2)  # Gradient magnitude
    Gphase = np.arctan2(Gy, Gx)    # Gradient phase (direction)

    # Normalize for visualization
    Gx = cv.normalize(Gx, None, 0, 255, cv.NORM_MINMAX)
    Gy = cv.normalize(Gy, None, 0, 255, cv.NORM_MINMAX)
    Gmag = cv.normalize(Gmag, None, 0, 255, cv.NORM_MINMAX)
    Gphase = cv.normalize(Gphase, None, 0, 255, cv.NORM_MINMAX)
    
    return Gx.astype(np.uint8), Gy.astype(np.uint8), Gmag.astype(np.uint8), Gphase.astype(np.uint8)

def AddGaussianNoise(img, sigma):
    """
    Adds independent Gaussian noise to a grayscale image.

    Parameters:
    - img: Input grayscale image (numpy array).
    - sigma: Standard deviation of the Gaussian noise.

    Returns:
    - Image with added noise.
    """
    # Generate Gaussian noise with mean 0 and standard deviation sigma
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    
    # Add the noise to the original image
    Ir = img.astype(np.float32) + noise

    # Clamp the values to be in the range [0, 255]
    Ir = np.clip(Ir, 0, 255).astype(np.uint8)
 
    return Ir

def AddSaltAndPepperNoise(img, salt_prob, pepper_prob):
    """
    Adds salt and pepper noise to a grayscale image.
    
    Parameters:
    - img: Input grayscale image (numpy array).
    - salt_prob: Probability of adding salt noise (white pixels).
    - pepper_prob: Probability of adding pepper noise (black pixels).
    
    Returns:
    - Noisy image with salt and pepper noise added.
    """
    noisy_img = img.copy()
    num_salt = np.ceil(salt_prob * img.size)
    num_pepper = np.ceil(pepper_prob * img.size)

    # Add salt noise
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    noisy_img[coords[0], coords[1]] = 255

    # Add pepper noise
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    noisy_img[coords[0], coords[1]] = 0

    return noisy_img

def AddPoissonNoise(img):
    """
    Adds Poisson noise to a grayscale image.
    
    Parameters:
    - img: Input grayscale image (numpy array).
    
    Returns:
    - Noisy image with Poisson noise added.
    """

    noisy_img = np.random.poisson(img).astype(np.uint8)

    return noisy_img

def AddUniformNoise(img, low, high):
    """
    Adds uniform noise to a grayscale image.
    
    Parameters:
    - img: Input grayscale image (numpy array).
    - low: Lower bound of the uniform distribution.
    - high: Upper bound of the uniform distribution.
    
    Returns:
    - Noisy image with uniform noise added.
    """
    noise = np.random.uniform(low, high, img.shape).astype(np.float32)
    noisy_img = img.astype(np.float32) + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img

def AddSpeckleNoise(img, sigma):
    """
    Adds speckle noise to a grayscale image.
    
    Parameters:
    - img: Input grayscale image (numpy array).
    - sigma: Standard deviation of the Gaussian noise.
    
    Returns:
    - Noisy image with speckle noise added.
    """
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy_img = img.astype(np.float32) + img.astype(np.float32) * noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img

def MedianFilter1(image, window_size):
    """
    Aplica un filtro de mediana a una imagen en escala de grises utilizando un histograma deslizante.

    Parámetros:
    - image: Imagen en escala de grises (numpy array).
    - window_size: Tamaño de la ventana cuadrada para calcular la mediana (debe ser impar).

    Retorna:
    - Imagen filtrada con el filtro de mediana.
    """

    # Asegurar que el tamaño de la ventana sea impar
    if window_size % 2 == 0:
        raise ValueError("El tamaño de la ventana debe ser un número impar.")

    half_w = window_size // 2  # Mitad de la ventana
    height, width = image.shape
    filtered_image = np.zeros_like(image, dtype=np.uint8)

    # Inicializar histograma para la primera ventana
    histogram = np.zeros(256, dtype=int)

    # Construir histograma inicial para la primera columna
    for y in range(window_size):
        for x in range(window_size):
            histogram[image[y, x]] += 1

    # Función para encontrar la mediana en el histograma acumulado
    def find_median(hist, total_pixels):
        count = 0
        for i in range(256):
            count += hist[i]
            if count >= total_pixels:
                return i

    median_pos = (window_size * window_size) // 2  # Posición del valor mediano

    # Aplicar filtro de mediana con histograma deslizante
    for y in range(height - window_size + 1):
        if y > 0:
            # Actualizar histograma eliminando la fila superior anterior y agregando la nueva fila inferior
            for x in range(window_size):
                histogram[image[y - 1, x]] -= 1  # Remover la fila superior
                histogram[image[y + window_size - 1, x]] += 1  # Agregar la fila inferior

        # Clonar histograma para manipularlo en la dirección X
        current_hist = histogram.copy()
        filtered_image[y + half_w, half_w] = find_median(current_hist, median_pos)

        for x in range(1, width - window_size + 1):
            # Deslizar ventana en la dirección X
            for i in range(window_size):
                current_hist[image[y + i, x - 1]] -= 1  # Quitar pixel de la izquierda
                current_hist[image[y + i, x + window_size - 1]] += 1  # Agregar pixel de la derecha

            # Calcular la mediana en la nueva ventana
            filtered_image[y + half_w, x + half_w] = find_median(current_hist, median_pos)

    return filtered_image

def MedianFilter(img, w):
    """
    Aplica un filtro de mediana a una imagen en escala de grises.

    Parámetros:
    - img: Imagen de entrada (numpy array en escala de grises).
    - w: Tamaño de la ventana cuadrada (debe ser impar).

    Retorna:
    - Ir: Imagen filtrada con el filtro de mediana.
    """
    height, width = img.shape
    half_w = w // 2  # Determina la mitad del ancho del filtro
    Ir = np.zeros_like(img)  # Imagen de salida

    # Recorremos cada píxel de la imagen
    for y in range(height):
        for x in range(width):
            # Definir los límites de la ventana de filtrado
            y1, y2 = max(0, y - half_w), min(height, y + half_w + 1)
            x1, x2 = max(0, x - half_w), min(width, x + half_w + 1)

            # Extraer la ventana y calcular la mediana
            window = img[y1:y2, x1:x2].flatten()
            med = np.median(window)

            # Asignar el valor mediano al píxel correspondiente
            Ir[y, x] = med

    return Ir

def NonLocalMeans(I, window_size, search_size, sigma):
    """
    Aplica el filtro de medios no locales (Non-Local Means) a una imagen en escala de grises.

    Parámetros:
    - I: Imagen en escala de grises (numpy array).
    - window_size: Tamaño del parche (debe ser impar).
    - search_size: Tamaño de la región de búsqueda (debe ser impar).
    - sigma: Parámetro de suavizado.

    Retorna:
    - Ir: Imagen suavizada.
    """
    I = I.astype(np.float32) / 255.0  # Normalizar la imagen
    height, width = I.shape
    half_w = window_size // 2
    half_s = search_size // 2
    Ir = np.zeros_like(I, dtype=np.float32)

    # Recorremos cada píxel de la imagen de salida
    for y in range(height):
        for x in range(width):
            val = 0
            norm = 0

            # Recorremos los píxeles dentro de la región de búsqueda
            for yr in range(max(0, y - half_s), min(height, y + half_s + 1)):
                for xr in range(max(0, x - half_s), min(width, x + half_s + 1)):
                    d = 0  # Distancia acumulada

                    # Comparar parches de tamaño window_size × window_size
                    for dy in range(-half_w, half_w + 1):
                        for dx in range(-half_w, half_w + 1):
                            y1, x1 = min(max(y + dy, 0), height - 1), min(max(x + dx, 0), width - 1)
                            y2, x2 = min(max(yr + dy, 0), height - 1), min(max(xr + dx, 0), width - 1)
                            d += (I[y1, x1] - I[y2, x2]) ** 2  # Diferencia entre ventanas

                    # Evitar desbordamientos limitando d
                    d = np.clip(d, 0, 5000)  

                    # Calcular peso basado en similitud
                    w = np.exp(-d / (2 * sigma ** 2))

                    # Acumular valores ponderados
                    val += w * I[yr, xr]
                    norm += w

            # Normalizar el resultado final
            Ir[y, x] = val / norm if norm > 0 else I[y, x]

    return (Ir * 255).astype(np.uint8)  # Convertir de vuelta a uint8

def BilateralFilter(I, ss, sr, niter):
    """
    Aplica un filtro bilateral iterativo a una imagen en escala de grises.

    Parámetros:
    - I: Imagen en escala de grises (numpy array).
    - ss: Desviación estándar del kernel espacial.
    - sr: Desviación estándar del kernel de rango.
    - niter: Número de iteraciones.

    Retorna:
    - Imagen filtrada con el filtro bilateral.
    """
    I = I.astype(np.float32) / 255.0  # Normalizar la imagen a rango [0,1]
    height, width = I.shape
    half_w = int(2.5 * ss)  # Definir el tamaño de la ventana

    for _ in range(niter):  # Aplicar iteraciones
        Ir = np.zeros_like(I)
        for y in range(height):
            for x in range(width):
                val = 0
                norm = 0
                
                for dy in range(-half_w, half_w + 1):
                    for dx in range(-half_w, half_w + 1):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            d2s = dx**2 + dy**2  # Distancia espacial al cuadrado
                            dr = I[y, x] - I[ny, nx]  # Diferencia de intensidad
                            
                            w = np.exp(-d2s / (2 * ss**2)) * np.exp(-dr**2 / (2 * sr**2))
                            val += w * I[ny, nx]
                            norm += w

                Ir[y, x] = val / norm if norm > 0 else I[y, x]

        I = Ir.copy()  # Copiar el resultado para la siguiente iteración

    return (I * 255).astype(np.uint8)  # Convertir de vuelta a uint8

def BilateralFilterFast(I, ss, sr, niter, n=3):
    """
    Aplica un filtro bilateral rápido a una imagen en escala de grises utilizando un coseno elevado.

    Parámetros:
    - I: Imagen en escala de grises (numpy array).
    - ss: Desviación estándar del kernel espacial.
    - sr: Desviación estándar del kernel de rango.
    - niter: Número de iteraciones.
    - n: Parámetro de aproximación (valor recomendado: n=3).

    Retorna:
    - Imagen filtrada con el filtro bilateral rápido.
    """
    I = I.astype(np.float32) / 255.0  # Normalizar imagen a [0,1]
    g = 1.0 / sr  # Factor de normalización
    height, width = I.shape

    for _ in range(niter):  # Iteraciones del filtro
        num = np.zeros_like(I, dtype=np.float32)
        den = np.zeros_like(I, dtype=np.float32)

        for i in range(n + 1):
            # Cálculo de coeficientes
            v = g * (2 * i - n) * I / np.sqrt(n)
            b = factorial(n) / (factorial(i) * factorial(n - i) * (2 ** (2 * n - 2)))

            # Crear imágenes Hi, Gi y Di (complejas)
            Hi = np.stack([np.cos(v), np.sin(v)], axis=-1)  # 2 canales: [cos(v), sin(v)]
            Gi = I[..., np.newaxis] * Hi  # Producto punto con I
            Di = Hi * b  # Aplicación del peso b

            # Aplicar suavizado Gaussiano separadamente en cada canal
            Gir = np.stack([gaussian_filter(Gi[..., 0], sigma=ss), 
                            gaussian_filter(Gi[..., 1], sigma=ss)], axis=-1)
            Hir = np.stack([gaussian_filter(Hi[..., 0], sigma=ss), 
                            gaussian_filter(Hi[..., 1], sigma=ss)], axis=-1)

            # Acumulación de numerador y denominador
            num += Di[..., 0] * Gir[..., 0] + Di[..., 1] * Gir[..., 1]
            den += Di[..., 0] * Hir[..., 0] + Di[..., 1] * Hir[..., 1]

        # División de los valores y tomar la parte real
        Ir = np.divide(num, den, out=I, where=den != 0)

        I = Ir.copy()  # Actualizar imagen para la siguiente iteración

    return (I * 255).astype(np.uint8)  # Convertir a uint8 (0-255)

def GaussianFilterGrayscale(img, sigma):
    """
    Applies a Gaussian filter to a grayscale image.

    Parameters:
    - img: Grayscale image (numpy array).
    - sigma: Standard deviation of the Gaussian filter.

    Returns:
    - Filtered image with the Gaussian filter applied.
    """
    img = img.astype(np.float32) / 255.0  # Normalize the image

    gauss = CreateGaussianKernel(sigma)
    img_smoothed = ConvolveSeparable(img, gauss, gauss)

    return (img_smoothed * 255).astype(np.uint8)  # Convert back to uint8

def GaussianFilterRGB(img, sigma):
    """
    Applies a Gaussian filter to an RGB image.

    Parameters:
    - img: RGB image (numpy array).
    - sigma: Standard deviation of the Gaussian filter.

    Returns:
    - Filtered image with the Gaussian filter applied.
    """
    # Verify that the image is RGB
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError("The image must be RGB.")

    # Convert the image to float32 type and normalize it
    img = img.astype(np.float32) / 255.0

    # Create the Gaussian kernel
    gauss = CreateGaussianKernel(sigma)

    # Initialize a smoothed image for the three channels
    img_smoothed = np.zeros_like(img)

    # Apply the Gaussian filter to each channel (R, G, B) independently
    for i in range(3):  # The three color channels: R, G, B
        img_smoothed[:, :, i] = ConvolveSeparable(img[:, :, i], gauss, gauss)

    # Convert the smoothed image back to uint8
    return (img_smoothed * 255).astype(np.uint8)

# thresholding methods

def RidlerCalvardThreshold(img, max_iterations=100, tolerance=1e-3):

    T_old = np.mean(img)
    
    for _ in range(max_iterations):
        G1 = img[img <= T_old]  
        G2 = img[img > T_old]
        
        if len(G1) == 0 or len(G2) == 0:
            break  
        
        mu1 = np.mean(G1)
        mu2 = np.mean(G2)
        T_new = (mu1 + mu2) / 2
        
        if abs(T_new - T_old) < tolerance:
            break
        
        T_old = T_new

        print(len(G1), len(G2), mu1, mu2, T_new, T_old)

    return T_old

def compute_histogram(I):
    """Calcula el histograma de la imagen en escala de grises."""
    hist = cv.calcHist([I], [0], None, [256], [0, 256])
    return hist.flatten()

def OtsuThreshold(I):
    """Implementación del algoritmo de Otsu para encontrar el umbral óptimo."""
    # 1. Calcular el histograma
    hist = compute_histogram(I)
    total_pixels = I.size
    
    # Inicialización de las medias m0 y m1
    m0 = 0
    m1 = np.sum(np.arange(256) * hist)
    w0 = 0
    w1 = np.sum(hist)
    
    # 2. Calcular el umbral óptimo
    max_between_class_variance = 0
    optimal_threshold = 0
    
    # Calcular la varianza intra-clase para cada umbral
    for t in range(1, 256):  # Iterar por todos los posibles umbrales (1 a 255)
        w0 += hist[t - 1]  # Proporción de píxeles en el grupo 0 (fondo)
        w1 -= hist[t - 1]  # Proporción de píxeles en el grupo 1 (objeto)
        
        # Calcular las medias de los grupos
        m0 += (t - 1) * hist[t - 1]
        m1 -= (t - 1) * hist[t - 1]
        
        # Si no hay píxeles en el fondo o en el objeto, continuar
        if w0 == 0 or w1 == 0:
            continue
        
        # Calcular la varianza entre clases
        mean_diff = m0 / w0 - m1 / w1
        between_class_variance = w0 * w1 * mean_diff ** 2
        
        # Comparar la varianza entre clases
        if between_class_variance > max_between_class_variance:
            max_between_class_variance = between_class_variance
            optimal_threshold = t
    
    return optimal_threshold