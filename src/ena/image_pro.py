import cv2 as cv
import numpy as np

def channels_bgr(img):
    """
    Splits the input BGR image into its individual blue, green, and red channels.

    Args:
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
        Args:
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

    Parameters:
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
        print("Error: channel images must have the same dimensions as the original image")
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
    blue, green, red = cv.split(img)
    zeros = np.zeros(img.shape[:2], dtype="uint8")
    
    red_green = cv.merge([zeros, green, red])
    red_blue = cv.merge([blue, zeros, red])
    green_blue = cv.merge([blue, green, zeros])
    
    return red_green, red_blue, green_blue

def histogram_equalization(img):
    # 1. Calcular el histograma
    hist, bins = np.histogram(img.flatten(), bins=256, range=[0, 256])
    
    # 2. Calcular la función de distribución acumulativa (CDF)
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]  # Normalizar el CDF para que vaya de 0 a 1
    
    # 3. Mapear los valores originales a los nuevos valores usando la CDF
    cdf_scaled = (cdf_normalized * 255).astype(np.uint8)  # Escalar a [0, 255]
    img_equalized = cdf_scaled[img]  # Usar la CDF como lookup table
    
    return img_equalized

def histogram_matching(img_1, img_2):
    # 1. Calcular el histograma de la imagen de referencia
    hist_ref, bins = np.histogram(img_1.flatten(), bins=256, range=[0, 256])
    
    # 2. Calcular el histograma de la imagen a ecualizar
    hist, bins = np.histogram(img_2.flatten(), bins=256, range=[0, 256])
    
    # 3. Calcular la función de distribución acumulativa (CDF) de ambas imágenes
    cdf_ref = hist_ref.cumsum()
    cdf_ref_normalized = cdf_ref / cdf_ref[-1]
    
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]
    
    # 4. Crear la tabla de correspondencia
    lookup_table = np.interp(cdf_normalized, cdf_ref_normalized, range(256)).astype(np.uint8)
    
    # 5. Aplicar la tabla de correspondencia a la imagen a ecualizar
    img_matched = lookup_table[img_2]
    
    return img_matched

def bgr_to_gray(img):
    blue, green, red = cv.split(img)
    gray_img = 0.299 * red + 0.587 * green + 0.114 * blue
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

    Args:
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

def show_two_images_side_by_side(img1, img2, title="Comparison", orientation="horizontal"):
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
    return Ir

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

de
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

    return Gx, Gy, G

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

def ComputeImageGradient(Img, sigma):
    """
    Compute the image gradient using the Gaussian derivative kernel.

    Parameters:
    - Img: The input image (numpy array).
    - sigma: The standard deviation of the Gaussian kernel.

    Returns:
    - Gx: The gradient image in the X direction.
    - Gy: The gradient image in the Y direction.
    - G: The magnitude of the gradient.
    """
    # Step 1: Create the Gaussian derivative kernel
    gauss_deriv = CreateGaussianDerivativeKernel(sigma)

    # Step 2: Apply the convolution
    Gx = ConvolveSeparable(Img, gauss_deriv, gauss_deriv)
    Gy = ConvolveSeparable(Img, gauss_deriv, gauss_deriv)

    # Step 3: Compute the magnitude and phase of the gradient

    # Step 3: Compute the magnitude and phase of the gradient
    Gmag = np.sqrt(Gx**2 + Gy**2)  # Gradient magnitude
    Gphase = np.arctan2(Gy, Gx)    # Gradient phase (direction)

    # Normalize for visualization
    Gx = cv.normalize(Gx, None, 0, 255, cv.NORM_MINMAX)
    Gy = cv.normalize(Gy, None, 0, 255, cv.NORM_MINMAX)
    Gmag = cv.normalize(Gmag, None, 0, 255, cv.NORM_MINMAX)
    Gphase = cv.normalize(Gphase, None, 0, 255, cv.NORM_MINMAX)
    

    return Gx, Gy, Gmag, Gphase