import cv2 as cv
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import factorial 
from collections import deque
from scipy.ndimage import convolve1d, zoom, sobel
from scipy.special import comb
from numba import njit


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

#  -----------------------------
#                             
#    Image Processing          
#    Histogram Functions      
#                              
#  -----------------------------

def CalHistogram(channel):
    
    """
    Compute the histogram of one-channel image.

    Parameters:
        channel (numpy.ndarray): The input channel image.

    Returns:
        numpy.ndarray: The computed histogram with 256 bins.
    """
    # Compute the histogram

    hist, bins = np.histogram(channel.flatten(), bins=256, range=[0, 256])
    
    return hist

def ClipHistogram(hist, clip_limit):
    """
    Clips the histogram by limiting each bin to the clip_limit and redistributes
    the clipped amount uniformly among all the bins.
    """
    excess = hist - clip_limit
    excess[excess < 0] = 0
    total_excess = np.sum(excess)
    
    # Clip the histogram
    hist = np.minimum(hist, clip_limit)
    
    # Redistribute the excess uniformly
    redist = total_excess // 256
    hist = hist + redist
    
    # Distribute the remainder sequentially
    remainder = total_excess % 256
    for i in range(256):
        if remainder <= 0:
            break
        hist[i] += 1
        remainder -= 1
        
    return hist

def CreateMapping(hist, block_size):
    """
    Calculates the mapping function (lookup table) from the clipped histogram
    using the cumulative distribution function (CDF).
    """
    cdf = np.cumsum(hist)
    # Avoid division by zero: find the first non-zero value
    cdf_min = cdf[np.nonzero(cdf)][0]
    
    # Normalize the CDF to [0, 255]
    mapping = np.round((cdf - cdf_min) / float(block_size - cdf_min) * 255).astype(np.uint8)
    return mapping

def ComputeMappings(image, n_rows, n_cols, cell_h, cell_w, clip_limit):
    """Computes histogram equalization mappings for each block."""
    mappings = [[None for _ in range(n_cols)] for _ in range(n_rows)]

    for i in range(n_rows):
        for j in range(n_cols):
            r0, r1 = i * cell_h, min((i + 1) * cell_h, image.shape[0])
            c0, c1 = j * cell_w, min((j + 1) * cell_w, image.shape[1])

            block = image[r0:r1, c0:c1]
            hist = CalHistogram(block)
            hist_clipped = ClipHistogram(hist, clip_limit)
            mappings[i][j] = CreateMapping(hist_clipped, block.size)

    return mappings

def ApplyInterpolation(image, mappings, n_rows, n_cols, cell_h, cell_w):
    """Applies bilinear interpolation to map pixel intensities using CLAHE mappings."""
    height, width = image.shape
    output = np.zeros_like(image, dtype=np.uint8)

    for y in range(height):
        i0, i1, y_weight = InterpolationIndices(y, cell_h, n_rows)
        
        for x in range(width):
            j0, j1, x_weight = InterpolationIndices(x, cell_w, n_cols)
            
            intensity = image[y, x]
            val_tl, val_tr = mappings[i0][j0][intensity], mappings[i0][j1][intensity]
            val_bl, val_br = mappings[i1][j0][intensity], mappings[i1][j1][intensity]

            top = val_tl * (1 - x_weight) + val_tr * x_weight
            bottom = val_bl * (1 - x_weight) + val_br * x_weight
            output[y, x] = int(np.round(top * (1 - y_weight) + bottom * y_weight))

    return output

def InterpolationIndices(coord, cell_size, max_index):
    """Computes interpolation indices and weights for bilinear interpolation."""
    f = (coord + 0.5) / cell_size - 0.5
    i0 = int(np.floor(f))
    i1 = min(i0 + 1, max_index - 1)
    weight = f - i0 if 0 <= i0 < max_index - 1 else 0
    return max(0, i0), i1, weight

def HistogramEqualizationGray(img):
    """
    Apply histogram equalization to an input image.
    
    Parameters:
        img (numpy.ndarray): Input grayscale image as a 2D numpy array.
    
    Returns:
        img: Image after applying histogram equalization, with the same shape as the input image.
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

def HistogramEqualizationRGB(image):
    """
    Apply histogram equalization to an RGB image by working on the luminance channel in YCrCb space.
    
    Parameters:
        image (numpy.ndarray): Input RGB image as a 3D numpy array.
    
    Returns:
        numpy.ndarray: Image after applying histogram equalization, with the same shape as the input image.
    """
    # Convertir la imagen al espacio de color YCrCb
    ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv.split(ycrcb)

    # Aplicar la equalización de histograma al canal de luminancia (Y)
    hist, bins = np.histogram(Y.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()  # Función de distribución acumulativa
    cdf_normalized = cdf / cdf[-1]  # Normalizar el CDF para que vaya de 0 a 1
    cdf_scaled = (cdf_normalized * 255).astype(np.uint8)  # Escalar a [0, 255]
    Y_equalized = cdf_scaled[Y]  # Usar la CDF como tabla de búsqueda

    # Reconstruir la imagen en el espacio YCrCb
    ycrcb_equalized = cv.merge([Y_equalized, Cr, Cb])

    # Convertir de vuelta al espacio de color BGR
    image_equalized = cv.cvtColor(ycrcb_equalized, cv.COLOR_YCrCb2BGR)

    return image_equalized

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

def HistogramEqualizationClaheGrayscale(image, clip_limit=10, grid_size=(8, 8)):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to a grayscale image 
    or just a single channel.

    Parameters:
        image: uint8 numpy array of shape (height, width).
        clip_limit: maximum allowed value for each histogram bin.
        grid_size: tuple (n_rows, n_cols) indicating how many blocks the image is divided into.

    Returns:
        equalized_image: uint8 numpy array of the resulting image with enhanced contrast.
    """
    height, width = image.shape
    n_rows, n_cols = grid_size
    cell_h, cell_w = height // n_rows, width // n_cols

    # Compute histogram mappings for each block
    mappings = ComputeMappings(image, n_rows, n_cols, cell_h, cell_w, clip_limit)

    # Apply bilinear interpolation to construct the output image
    return ApplyInterpolation(image, mappings, n_rows, n_cols, cell_h, cell_w)

def HistogramEqualizationClaheRGB(image, clip_limit=10, grid_size=(8, 8)):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to an RGB image.
    
    Parameters:
        image: uint8 numpy array of shape (height, width, 3) representing an RGB image.
        clip_limit: Maximum allowed value for each histogram bin.
        grid_size: Tuple (n_rows, n_cols) indicating the number of regions the image is divided into.
        
    Returns:
        equalized_image: uint8 numpy array of the resulting image with enhanced contrast.
    """
    # Convert image to LAB color space
    lab_image = cv.cvtColor(image, cv.COLOR_RGB2LAB)
    
    # Extract the L channel (brightness)
    l_channel, a_channel, b_channel = cv.split(lab_image)
    
    # Apply CLAHE to the L channel
    clahe = HistogramEqualizationClaheGrayscale(l_channel, clip_limit, grid_size)
    #l_channel_eq = clahe.apply(l_channel)
    l_channel_eq = clahe
    
    # Merge the modified L channel back with A and B channels
    lab_eq = cv.merge((l_channel_eq, a_channel, b_channel))
    
    # Convert the image back to RGB color space
    equalized_image = cv.cvtColor(lab_eq, cv.COLOR_LAB2RGB)
    
    return equalized_image

def BihistogramEqualizationRGB(image):
    """
    Applies Bi-Histogram Equalization to a color image without distorting colors.

    This function enhances the contrast of an image by applying histogram equalization 
    separately to two regions: pixels with luminance values lower than a threshold and 
    those with higher values. The threshold is determined using Otsu's method.

    Parameters:
        image (numpy.ndarray): Input color image in BGR format.

    Returns:
        numpy.ndarray: The contrast-enhanced image in BGR format.
    """
    # Convert to YCrCb to work only on the luminance channel
    ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv.split(ycrcb)

    # Step 1: Compute the threshold X_T using Otsu's method
    X_T = OtsuThreshold(Y)
    print(X_T)

    # Step 2: Separate the two histograms (below and above X_T)
    mask_L = Y <= X_T  # Mask for the lower region
    mask_U = Y > X_T   # Mask for the upper region

    # Step 3: Compute the cumulative distribution function (CDF) for the entire luminance channel
    hist, bins = np.histogram(Y.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()  # Cumulative distribution function
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())  # Normalize to [0, 255]
    cdf_normalized = np.round(cdf_normalized).astype(np.uint8)

    # Step 4: Adjust the lower and upper regions using the CDF
    Y_eq = np.copy(Y)
    Y_eq[mask_L] = cdf_normalized[Y[mask_L]]
    Y_eq[mask_U] = cdf_normalized[Y[mask_U]]

    # Merge Y, Cr, and Cb channels and convert back to BGR
    result = cv.merge([Y_eq, Cr, Cb])
    result = cv.cvtColor(result, cv.COLOR_YCrCb2BGR)

    return result

def BihistogramEqualizationGrayscale(image):
    """
    Applies Bi-Histogram Equalization to a grayscale image.

    This function enhances the contrast of a grayscale image by applying histogram equalization 
    separately to two regions: pixels with intensity values lower than a threshold and those with 
    higher values. The threshold is determined using the median of the image.

    Parameters:
        image (numpy.ndarray): Input grayscale image.

    Returns:
        numpy.ndarray: The contrast-enhanced grayscale image.

    Raises:
        ValueError: If the input image is not in grayscale format.
    """
    # Validate that the image is grayscale
    if len(image.shape) != 2:
        raise ValueError("The input image must be in grayscale format.")

    # Step 1: Compute the threshold X_T as the median of the image
    X_T = np.median(image)

    # Step 2: Separate the two histograms (below and above X_T)
    mask_L = image <= X_T  # Mask for the lower region
    mask_U = image > X_T   # Mask for the upper region

    # Step 3: Compute the cumulative distribution function (CDF) for the entire image
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()  # Cumulative distribution function
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())  # Normalize to [0, 255]
    cdf_normalized = np.round(cdf_normalized).astype(np.uint8)

    # Step 4: Adjust the lower and upper regions using the CDF
    image_eq = np.copy(image)
    image_eq[mask_L] = cdf_normalized[image[mask_L]]
    image_eq[mask_U] = cdf_normalized[image[mask_U]]

    # Ensure values are within the range [0, 255]
    image_eq = np.clip(image_eq, 0, 255).astype(np.uint8)

    return image_eq

# ---------------------------------
#                                  #
#   Image Negative                 #
#                                  #

def ImageNegative(img):
    """
    Compute the negative of an image by subtracting each pixel value from 255.

    Parameters:
        img (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Negative of the input image.
    """

    return 255 - img

def GammaCorrection(image, gamma, intensity_levels=256):
    """
    Applies gamma correction to a grayscale image.

    Parameters:
        image: uint8 numpy array of shape (height, width).
        gamma: Gamma value for correction (gamma > 1 darkens, gamma < 1 brightens).
        intensity_levels: Number of intensity levels (default is 256 for 8-bit images).

    Returns:
        corrected_image: uint8 numpy array with gamma correction applied.
    """
    # Normalizar intensidades al rango [0,1]
    normalized = image / (intensity_levels - 1)
    
    # Aplicar transformación gamma: s = (r ^ gamma) * (L - 1)
    corrected = (normalized ** gamma) * (intensity_levels - 1)
    
    # Convertir de nuevo a uint8
    return corrected.astype(np.uint8)

def BgrToGray(img):
    """
    Convert a BGR image to a grayscale image.
    This function takes an image in BGR format and converts it to a grayscale image
    using a weighted sum of the color channels based on human visual perception.

    Parameters:
        - img (numpy.ndarray): Input image in BGR format.

    Returns:
        - numpy.ndarray: Grayscale image.
    """

    # Convert to float for precise calculation (avoiding integer rounding issues)
    img = img.astype(np.float32)

    # Split image into individual color channels
    blue, green, red = cv.split(img)

    # Weighted average to convert to grayscale:
    # The weights reflect the human eye's sensitivity to each color.
    # Humans are most sensitive to green, then red, and least to blue.
    # This formula is based on the ITU-R BT.601 standard:
    #    Gray = 0.299 * R + 0.587 * G + 0.114 * B
    gray_img = 0.299 * red + 0.587 * green + 0.114 * blue

    # Ensure pixel values remain in the valid 0-255 range
    gray_img = np.clip(gray_img, 0, 255)

    # Convert back to 8-bit integer format
    return gray_img.astype(np.uint8)

# ---------------------------------
#                                   #
#   lebaling functions an connected #
#   components                      #
#                                   #
# ---------------------------------

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

def FloodFillDFS(img, seed, new_color):
    """
    Flood Fill Algorithm using Depth-First Search (DFS).

    Parameters:
    - img: np.ndarray -> A grayscale or binary image (pixel matrix).
    - seed: tuple -> Coordinates (x, y) of the seed pixel.
    - new_color: int -> The color that the connected region will be filled with.

    This function modifies the original image by changing the color of the region connected to the seed pixel.
    """
    
    x, y = seed
    orig_color = img[y, x]  # Get the original color of the seed pixel
    
    if orig_color == new_color:
        return  # If the pixel already has the new color, do nothing

    frontier = [seed]  # Initialize the stack with the seed pixel
    img[y, x] = new_color  # Color the seed pixel with the new color

    # Directions to explore the 4 neighbors (right, left, down, up)
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while frontier:
        cx, cy = frontier.pop()  # Pop the last pixel from the stack (LIFO)

        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy  # Coordinates of the neighbor

            # Check if the neighbor is within bounds and has the original color
            if 0 <= nx < img.shape[1] and 0 <= ny < img.shape[0] and img[ny, nx] == orig_color:
                frontier.append((nx, ny))  # Add the neighbor to the stack
                img[ny, nx] = new_color  # Change its color

                # Visualize the process (optional)
                cv.imshow('Flood Fill', img)
                cv.waitKey(1)

def FloodFillBFS(img: np.ndarray, seed: tuple, new_color: int):
    """
    Flood Fill algorithm using Breadth-First Search (BFS).
    
    Parameters:
    - img: np.ndarray -> Image in grayscale or binary (pixel matrix).
    - seed: tuple -> Coordinates (x, y) of the seed pixel.
    - new_color: int -> The color to fill the region with.
    
    Modifies the original image by replacing the connected region starting from the seed pixel.
    """
    
    x, y = seed
    orig_color = img[y, x]  # Get the original color of the seed pixel
    
    if orig_color == new_color:
        return  # If the seed already has the new color, do nothing
    
    queue = deque([(x, y)])  # Initialize the queue with the seed pixel
    img[y, x] = new_color  # Color the seed pixel with the new color

    # Directions to explore the 4 neighbors (right, left, down, up)
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while queue:
        cx, cy = queue.popleft()  # Pop the first pixel from the queue (FIFO)

        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy  # Coordinates of the neighbor

            # Check if the neighbor is within the image bounds and has the original color
            if 0 <= nx < img.shape[1] and 0 <= ny < img.shape[0] and img[ny, nx] == orig_color:
                queue.append((nx, ny))  # Add the neighbor to the queue
                img[ny, nx] = new_color  # Change its color

                 #Visualization of the process (optional)
                cv.imshow('Flood Fill BFS', img)
                cv.waitKey(1)

class UnionFind:
    """
    Union-Find data structure with path compression.
    
    This structure efficiently manages disjoint sets, commonly used for connected component labeling.
    It supports two main operations:
    - `find(x)`: Finds the representative (root) of the set containing `x`, applying path compression.
    - `union(x, y)`: Merges the sets containing `x` and `y`, maintaining a hierarchical structure.
    """
    def __init__(self, size):
        """
        Initializes the Union-Find structure.
        
        Parameters:
        size (int): The total number of elements (typically the number of pixels in the image).
        """
        self.parent = np.arange(size)  # Each element starts as its own parent
    
    def find(self, x):
        """
        Finds the representative element (root) of the set containing `x`, applying path compression.
        
        Parameters:
        x (int): The element whose root is being searched.

        Returns:
        int: The root of the set containing `x`.
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression for efficiency
        return self.parent[x]
    
    def union(self, x, y):
        """
        Merges the sets containing `x` and `y` by linking their roots.

        Parameters:
        x (int): An element from the first set.
        y (int): An element from the second set.
        """
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_y] = root_x  # Attach root_y to root_x

def ConnectedComponentsByUnionFind(image):
    """
    Performs connected components labeling on a binary image using the Union-Find algorithm.
    
    This method assigns unique labels to connected regions in a binary image. It follows a 
    two-pass approach: the first pass assigns preliminary labels and records equivalences, 
    while the second pass resolves these equivalences using the Union-Find structure.

    Parameters:
    image (numpy.ndarray): A 2D binary image where non-zero pixels represent foreground objects 
                           and zero pixels represent the background.

    Returns:
    numpy.ndarray: A 2D array of the same shape as the input image, where each connected component 
                   is assigned a unique integer label.
    """
    height, width = image.shape
    label_image = np.zeros((height, width), dtype=int)  # Output label array
    uf = UnionFind(height * width)  # Union-Find structure for label merging
    next_label = 1  # Initial label counter

    # First pass: Preliminary labeling and equivalence registration
    for y in range(height):
        for x in range(width):
            if image[y, x] == 0:  # Skip background pixels
                continue
            
            neighbors = []
            if x > 0 and image[y, x] == image[y, x - 1]:  # Left neighbor
                neighbors.append(label_image[y, x - 1])
            if y > 0 and image[y, x] == image[y - 1, x]:  # Upper neighbor
                neighbors.append(label_image[y - 1, x])
            
            if neighbors:
                min_label = min(neighbors)
                label_image[y, x] = min_label
                for label in neighbors:
                    uf.union(min_label, label)
            else:
                label_image[y, x] = next_label
                next_label += 1

    # Second pass: Resolve equivalences and reassign final labels
    labels_flat = label_image.flatten()
    labels_flat = np.vectorize(uf.find)(labels_flat)  # Apply find operation across all labels
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

# ------------------------------------
#                                     #
#   Convolution functions and kernels #
#                                     #
# ------------------------------------

def CreateLanczosLowpassFilter(shape, cutoff_frequency, a=3):
    """
    Creates a Lanczos low-pass filter kernel in the frequency domain.

    Parameters:
        shape : tuple
            Shape of the filter (rows, cols), typically matching the image dimensions.
        cutoff_frequency : float
            Frequency scaling factor (controls sharpness).
        a : int
            Lanczos window parameter (commonly 2 or 3). Larger values = narrower main lobe.

    Returns:
        filter_kernel : (np.ndarray)
            Lanczos low-pass filter kernel as a 2D numpy array.
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2

    # Coordenadas relativas al centro
    Y, X = np.ogrid[:rows, :cols]
    dx = X - ccol
    dy = Y - crow
    radius = np.sqrt(dx**2 + dy**2)

    # Normalizar el radio para hacerlo compatible con el parámetro a
    x = (radius / cutoff_frequency).astype(np.float32)

    # sinc(x) = sin(pi x) / (pi x), definida como 1 en x = 0
    def sinc(z):
        z = np.where(z == 0, 1e-8, z)  # evitar división por cero
        return np.sin(np.pi * z) / (np.pi * z)

    # Kernel de Lanczos en 2D: sinc(x) * sinc(x/a)
    lanczos_kernel = sinc(x) * sinc(x / a)

    # Forzar ceros fuera de la ventana a
    lanczos_kernel[x > a] = 0

    return lanczos_kernel.astype(np.float32)

def CreateButterworthLowpassFilter(shape, cutoff_frequency, order):
    """
    Creates a Butterworth low-pass filter kernel in the frequency domain.

    Parameters:
        shape : tuple
            Shape of the filter (rows, cols), typically matching the image dimensions.
        cutoff_frequency : float
            Cutoff frequency for the low-pass filter.
        order : int
            Order of the Butterworth filter, controlling the sharpness of the transition.

    Returns:
        filter_kernel : np.ndarray
            Butterworth low-pass filter kernel as a 2D numpy array.
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    distance = np.sqrt((X - ccol)**2 + (Y - crow)**2)
    
    # Butterworth low-pass filter formula
    filter_kernel = 1 / (1 + (distance / (cutoff_frequency + 1e-5))**(2 * order))  # Avoid division by zero
    return filter_kernel.astype(np.float32)

def CreateGaussianLowpassFilter(shape, cutoff_frequency):
    """
    Creates a Gaussian low-pass filter kernel in the frequency domain.

    Parameters:
        shape : tuple
            Shape of the filter (rows, cols), typically matching the image dimensions.
        cutoff_frequency : float
            Cutoff frequency for the low-pass filter.

    Returns:
        filter_kernel : np.ndarray
            Gaussian low-pass filter kernel as a 2D numpy array.
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    distance = np.sqrt((X - ccol)**2 + (Y - crow)**2)
    
    # Gaussian low-pass filter formula
    filter_kernel = np.exp(-(distance**2) / (2 * (cutoff_frequency**2)))
    return filter_kernel.astype(np.float32)

def CreateIdealLowpassFilter(shape, cutoff_frequency):
    """
    Creates an ideal low-pass filter kernel in the frequency domain.

    Parameters:
        shape : (tuple) Shape of the filter (rows, cols), typically matching the image dimensions.
        cutoff_frequency : (float) Cutoff frequency for the low-pass filter.

    Returns:
        filter_kernel : (np.ndarray) Ideal low-pass filter kernel as a 2D numpy array.
    """

    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    distance = np.sqrt((X - ccol)**2 + (Y - crow)**2)
    
    # Ideal low-pass filter formula
    filter_kernel = distance <= cutoff_frequency
    return filter_kernel.astype(np.float32)

def ConvolveSeparable(I, gh, gv):
    """
    Applies separable convolution to the input image I using 1D kernels gh (horizontal) and gv (vertical).

    Parameters:
    - I: 2D numpy array representing the input grayscale image.
    - gh: 1D numpy array representing the horizontal convolution kernel.
    - gv: 1D numpy array representing the vertical convolution kernel.

    Returns:
    - Itmp: 2D numpy array representing the image after applying both convolutions.
            Note: The returned image is the intermediate result after vertical convolution,
            with padding removed.
    """

    height, width = I.shape
    w = len(gh)
    pad_size = w // 2

    # Step 1: Apply padding to the input image
    I_padded = np.pad(I, ((pad_size, pad_size), (pad_size, pad_size)), mode='reflect')

    # Step 2: Horizontal convolution
    Itmp = np.zeros_like(I_padded, dtype=np.float32)
    for y in range(height):
        for x in range(width):
            val = 0
            for i in range(w):
                val += gh[i] * I_padded[y + pad_size, x + i]
            Itmp[y + pad_size, x + pad_size] = val

    # Step 3: Vertical convolution
    Ir = np.zeros_like(I_padded, dtype=np.float32)
    for y in range(height):
        for x in range(width):
            val = 0
            for i in range(w):
                val += gv[i] * Itmp[y + i, x + pad_size]
            Ir[y + pad_size, x + pad_size] = val

    # Remove padding (currently only returning the intermediate result)
    Ir = Ir[pad_size:-pad_size, pad_size:-pad_size]
    return Ir
    
def ConvolveSeparableOpt(I, gh, gv):
    """
    Applies separable convolution to the input image I using 1D kernels gh (horizontal) and gv (vertical).
    
    Optimized version using scipy.ndimage.convolve1d.

    Parameters:
    - I: 2D numpy array (grayscale image).
    - gh: 1D numpy array (horizontal kernel).
    - gv: 1D numpy array (vertical kernel).

    Returns:
    - Ir: 2D numpy array (result after separable convolution).
    """
    # Apply horizontal convolution (axis=1 -> along columns)
    Itmp = convolve1d(I, gh, axis=1, mode='reflect')

    # Apply vertical convolution (axis=0 -> along rows)
    Ir = convolve1d(Itmp, gv, axis=0, mode='reflect')

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

def CreateGaussianSecondDerivativeKernel(sigma):
    """
    Generates a 1D Gaussian second derivative kernel for edge detection.

    Parameters:
    - sigma (float): Standard deviation of the Gaussian function.

    Returns:
    - gauss_deriv2 (numpy.ndarray): The 1D Gaussian second derivative kernel.
    """

    # Step 1: Get the half-width of the kernel based on sigma
    half_width = GetKernelHalfWidth(sigma)

    # Step 2: Calculate the total width (always odd)
    w = 2 * half_width + 1

    # Step 3: Initialize the kernel and normalization factor
    gauss_deriv2 = np.zeros(w, dtype=np.float32)
    norm = 0.0

    # Step 4: Compute the second derivative of the Gaussian function for each position
    for i in range(w):
        x = i - half_width  # Center the kernel at zero
        # Compute the second derivative of the Gaussian at position x
        gauss_deriv2[i] = (x ** 2 / (sigma ** 4) - 1 / (sigma ** 2)) * np.exp(- (x ** 2) / (2 * sigma ** 2))
        # Accumulate the absolute value for normalization
        norm += abs(gauss_deriv2[i])

    # Step 5: Normalize the kernel so that the sum of absolute values is 1
    gauss_deriv2 /= norm

    return gauss_deriv2

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

    Gphase = np.arctan2(Gy, Gx)  # Fase del gradiente (dirección)

    # Normalizar para visualización
    Gx = cv.normalize(Gx, None, 0, 255, cv.NORM_MINMAX)
    Gy = cv.normalize(Gy, None, 0, 255, cv.NORM_MINMAX)
    G = cv.normalize(G, None, 0, 255, cv.NORM_MINMAX)
    Gphase = cv.normalize(Gphase, None, 0, 255, cv.NORM_MINMAX)

    return Gx.astype(np.uint8), Gy.astype(np.uint8), G.astype(np.uint8) , Gphase.astype(np.uint8)

def ComputeLaplacianGaussian(img, sigma_s, sigma_d):
    """
    Compute the Laplacian of Gaussian using separable convolution.

    Parameters:
    - img: Input image (grayscale, float32).
    - sigma_s: sigma for smoothing Gaussian.
    - sigma_d: sigma for second derivative.

    Returns:
    - LoG: Laplacian of Gaussian image.
    """
    img = img.astype(np.float32)

    gauss = CreateGaussianKernel(sigma_s)
    gauss_2nd = CreateGaussianSecondDerivativeKernel(sigma_d)

    Gxx = ConvolveSeparableOpt(img, gauss_2nd, gauss)  # ∂²I/∂x²
    Gyy = ConvolveSeparableOpt(img, gauss, gauss_2nd)  # ∂²I/∂y²

    LoG = Gxx + Gyy  # Laplacian = d²/dx² + d²/dy²

    LoG = cv.normalize(LoG, None, 0, 255, cv.NORM_MINMAX)
    return LoG.astype(np.uint8)

def ComputeImageGradient(img, sigma_s, sigma_d):
    """
    Compute the image gradient using the Gaussian derivative kernel.

    Parameters:
    - img: The input image (numpy array).
    - sigma_s: Standard deviation for the Gaussian kernel.
    - sigma_d: Standard deviation for the Gaussian derivative kernel.

    Returns:
    - Gx: The gradient image in the X direction.
    - Gy: The gradient image in the Y direction.
    - Gmag: The magnitude of the gradient.
    - Gphase: The phase of the gradient (direction).
    """
    # convertir a 32 bits y normalizar
    
    img = img.astype(np.float32)

    # Step 1: Create the Gaussian derivative kernel
    gauss = CreateGaussianKernel(sigma_s)
    gauss_deriv = CreateGaussianDerivativeKernel(sigma_d)

    Gx = ConvolveSeparableOpt(img, gauss_deriv, gauss)  # Convolve with the Gaussian derivative kernel
    Gy = ConvolveSeparableOpt(img, gauss, gauss_deriv)  # Convolve with the Gaussian kernel
   
    # Step 3: Compute the magnitude and phase of the gradient

    # Step 3: Compute the magnitude and phase of the gradient
    Gmag = np.sqrt(Gx**2 + Gy**2)  # Gradient magnitude
    Gphase = np.arctan2(Gy, Gx)    # Gradient phase (direction)
    #guarda en archivo la magnitud con numpa

    # Normalize for visualization
    Gx = cv.normalize(Gx, None, 0, 255, cv.NORM_MINMAX)
    Gy = cv.normalize(Gy, None, 0, 255, cv.NORM_MINMAX)
    Gmag = cv.normalize(Gmag, None, 0, 255, cv.NORM_MINMAX)
    Gphase = cv.normalize(Gphase, None, 0, 255, cv.NORM_MINMAX)
    
    return Gx.astype(np.uint8), Gy.astype(np.uint8), Gmag.astype(np.uint8), Gphase.astype(np.uint8)

# ---------------------------------
#                                   #
#   Noise functions                #
#                                   #
# ---------------------------------

def AddPeriodicNoise(img, amplitude=30, frequency=0.05, angle=0):
    """
    Adds periodic (sinusoidal) noise to a grayscale or color image.

    Parameters:
    - img: Input image (numpy array, uint8, grayscale or color).
    - amplitude: Amplitude of the sinusoidal noise.
    - frequency: Frequency of the sinusoidal pattern (cycles per pixel).
    - angle: Angle of the sinusoidal pattern in degrees.

    Returns:
    - Image with periodic noise.
    """

    # Prepare grid
    rows, cols = img.shape[:2]
    yy, xx = np.mgrid[0:rows, 0:cols]
    theta = np.deg2rad(angle)
    # Create sinusoidal pattern
    sinusoid = amplitude * np.sin(2 * np.pi * frequency * (xx * np.cos(theta) + yy * np.sin(theta)))

    # If image is color, add noise to all channels
    if img.ndim == 3:
        noisy = img.astype(np.float32) + sinusoid[..., None]
    else:
        noisy = img.astype(np.float32) + sinusoid

    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

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

# --------------------------------- 
#                                   #
#    Filter functions              #
#                                   #
# ---------------------------------

def ApplyFrequencyDomainFilter(image, kernel):
    """
    Applies a frequency domain filter to a grayscale image.

    This function computes the 2D Fourier Transform of the input image, applies the given filter 
    in the frequency domain, and then performs the inverse Fourier Transform to return the filtered image.

    Parameters:
        image : np.ndarray
            Input grayscale image (2D numpy array).

        kernel : np.ndarray
            Frequency domain filter (2D numpy array) with the same shape as the input image.

    Returns:
        filtered_image : np.ndarray
            Filtered image (uint8) normalized to the range [0, 255].

    Notes:
    -----
    - The input image is assumed to be in grayscale format.
    - The kernel should be designed in the frequency domain and have the same dimensions as the input image.
    - The output image is normalized to ensure proper visualization.
    """
    # Compute the 2D Fourier Transform of the image
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)  # Shift zero frequency to the center

    # Apply the filter in the frequency domain
    filtered_freq = fshift * kernel

    # Compute the inverse Fourier Transform to return to the spatial domain
    temp = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_freq)))

    # Normalize the result to the range [0, 255] and convert to uint8
    filtered_image = cv.normalize(temp, None, 0, 255, cv.NORM_MINMAX)
    filtered_image = np.uint8(filtered_image)

    return filtered_image



def MeanShiftFilter(I, hs, hr, max_iter=10, epsilon=1e-3):
    """
    Optimized Mean Shift filter for grayscale images (edge-preserving smoothing).

    Parameters:
    - I: Input grayscale image (numpy array, uint8 or float32).
    - hs: Spatial bandwidth (controls spatial window size).
    - hr: Range bandwidth (controls intensity similarity).
    - max_iter: Maximum number of iterations per pixel.
    - epsilon: Convergence threshold for the mean-shift vector.

    Returns:
    - Ir: Output image after mean-shift filtering (same shape as I).
    """
    I = I.astype(np.float32)
    height, width = I.shape
    Ir = np.zeros_like(I, dtype=np.float32)
    hs = int(np.ceil(hs))

    # Precompute spatial grid for window
    grid_y, grid_x = np.mgrid[-hs:hs+1, -hs:hs+1]
    spatial_dist2 = (grid_x**2 + grid_y**2) / (hs**2)

    for y in range(height):
        for x in range(width):
            xr, yr, vr = x, y, I[y, x]
            for _ in range(max_iter):
                # Define window bounds
                y0 = max(0, int(round(yr)) - hs)
                y1 = min(height, int(round(yr)) + hs + 1)
                x0 = max(0, int(round(xr)) - hs)
                x1 = min(width, int(round(xr)) + hs + 1)

                patch = I[y0:y1, x0:x1]
                gy, gx = np.ogrid[y0-y:int(y1-y), x0-x:int(x1-x)]
                # Adjust spatial distances for border windows
                sy = grid_y[(gy + hs)[:patch.shape[0]], (gx + hs)[:patch.shape[1]]]
                sx = grid_x[(gy + hs)[:patch.shape[0]], (gx + hs)[:patch.shape[1]]]
                spatial = (sx**2 + sy**2) / (hs**2)

                range_dist2 = ((patch - vr) ** 2) / (hr ** 2)
                w = np.exp(-0.5 * (spatial + range_dist2))

                norm = np.sum(w)
                if norm == 0:
                    break
                x_new = np.sum(w * np.arange(x0, x1)) / norm
                y_new = np.sum(w * np.arange(y0, y1)[:, None]) / norm
                v_new = np.sum(w * patch) / norm

                shift = np.sqrt((xr - x_new) ** 2 + (yr - y_new) ** 2 + (vr - v_new) ** 2)
                xr, yr, vr = x_new, y_new, v_new
                if shift < epsilon:
                    break
            Ir[y, x] = vr

    return np.clip(Ir, 0, 255).astype(np.uint8)

def MedianFilterGrayscale(image, window_size):
    """
    Applies a median filter to a grayscale image using efficient NumPy operations.

    Parameters:
    - image: Grayscale image (numpy array).
    - window_size: Size of the square window to compute the median (must be odd).

    Returns:
    - Filtered image with the median filter applied.
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd.")

    pad = window_size // 2
    # Pad the image to handle borders
    padded = np.pad(image, pad, mode='edge')
    height, width = image.shape
    filtered = np.zeros_like(image, dtype=np.uint8)

    # Use a sliding window and compute the median for each region
    for y in range(height):
        for x in range(width):
            window = padded[y:y+window_size, x:x+window_size]
            filtered[y, x] = np.median(window)

    return filtered

def MedianFilterBGR(image, window_size):
    """
    Applies a median filter to a BGR color image using efficient NumPy operations.

    Parameters:
    - image: BGR color image (numpy array, shape HxWx3).
    - window_size: Size of the square window to compute the median (must be odd).

    Returns:
    - Filtered image with the median filter applied to each channel.
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd.")

    pad = window_size // 2
    height, width, channels = image.shape
    filtered = np.zeros_like(image, dtype=np.uint8)

    # Apply the median filter to each channel independently
    for c in range(channels):
        padded = np.pad(image[:, :, c], pad, mode='edge')
        for y in range(height):
            for x in range(width):
                window = padded[y:y+window_size, x:x+window_size]
                filtered[y, x, c] = np.median(window)
    return filtered

def MeanShiftFilterBGR(image, hs, hr, max_iter=5, eps=1.0):
    """
    Fast mean-shift filter for BGR images (applied in LAB space).

    Parameters:
        image: Input BGR image (uint8).
        hs: Spatial bandwidth.
        hr: Range (color) bandwidth.
        max_iter: Maximum iterations.
        eps: Convergence threshold.

    Returns:
        Filtered BGR image (uint8).
    """
    # Convert BGR to LAB
    lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    lab_filtered = np.zeros_like(lab)
    # Apply mean-shift to each channel independently (L, A, B)
    for c in range(3):
        lab_filtered[..., c] = MeanShiftFilterGrayscale1(lab[..., c], hs, hr, max_iter, eps)
    # Convert back to BGR
    return cv.cvtColor(lab_filtered, cv.COLOR_LAB2BGR)

def MeanShiftFilterGrayscale1(image, hs, hr, max_iter=5, eps=1.0):
    """
    Mean-Shift filtering para imágenes en escala de grises.
    """
    h, w = image.shape
    image_f = image.astype(np.float32)
    output = np.zeros_like(image_f)

    # Precalcular kernel espacial (gaussiano)
    kernel_size = 2 * hs + 1
    y_idx = np.arange(kernel_size) - hs
    x_idx = np.arange(kernel_size) - hs
    y_grid, x_grid = np.meshgrid(y_idx, x_idx, indexing='ij')
    spatial_kernel = np.exp(-(x_grid**2 + y_grid**2) / (2 * hs**2)).astype(np.float32)

    for i in range(h):
        for j in range(w):
            xc, yc, vc = j, i, image_f[i, j]
            for _ in range(max_iter):
                # Calcular bordes de la ventana
                x_min = max(int(xc - hs), 0)
                x_max = min(int(xc + hs + 1), w)
                y_min = max(int(yc - hs), 0)
                y_max = min(int(yc + hs + 1), h)

                window = image_f[y_min:y_max, x_min:x_max]

                # Extraer la porción correspondiente del kernel espacial
                sk_y0 = int(hs - (yc - y_min))
                sk_x0 = int(hs - (xc - x_min))
                sk_y1 = sk_y0 + window.shape[0]
                sk_x1 = sk_x0 + window.shape[1]
                sk = spatial_kernel[sk_y0:sk_y1, sk_x0:sk_x1]

                # --- Asegurar shapes iguales ---
                if sk.shape != window.shape:
                    min_shape = (min(sk.shape[0], window.shape[0]), min(sk.shape[1], window.shape[1]))
                    sk = sk[:min_shape[0], :min_shape[1]]
                    window = window[:min_shape[0], :min_shape[1]]
                # --------------------------------

                # Calcular pesos rango y totales (vectorizado)
                range_weights = np.exp(-((window - vc) ** 2) / (2 * hr * hr))
                weights = sk * range_weights
                total_weight = np.sum(weights)
                if total_weight < 1e-5:
                    break

                # Calcular nueva media ponderada (vectorizado)
                wy = np.arange(y_min, y_min + window.shape[0])[:, None]
                wx = np.arange(x_min, x_min + window.shape[1])[None, :]

                mean_x = np.sum(weights * wx) / total_weight
                mean_y = np.sum(weights * wy) / total_weight
                mean_v = np.sum(weights * window) / total_weight

                shift = np.sqrt((mean_x - xc) ** 2 + (mean_y - yc) ** 2 + (mean_v - vc) ** 2)
                xc, yc, vc = mean_x, mean_y, mean_v

                if shift < eps:
                    break

            output[i, j] = vc

    return np.clip(output, 0, 255).astype(np.uint8)

@njit
def MeanShiftFilterGrayscale(image, hs, hr, max_iter=5, eps=1.0):
    h, w = image.shape
    image_f = image.astype(np.float32)
    output = np.zeros_like(image_f)

    kernel_size = 2 * hs + 1
    spatial_kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    for i in range(kernel_size):
        for j in range(kernel_size):
            dy = i - hs
            dx = j - hs
            spatial_kernel[i, j] = np.exp(-(dx**2 + dy**2) / (2 * hs**2))

    for i in range(h):
        for j in range(w):
            xc, yc = j, i
            vc = image_f[i, j]
            for _ in range(max_iter):
                x_min = int(max(xc - hs, 0))
                x_max = int(min(xc + hs + 1, w))
                y_min = int(max(yc - hs, 0))
                y_max = int(min(yc + hs + 1, h))
                window = image_f[y_min:y_max, x_min:x_max]

                sk_y0 = int(hs - (yc - y_min))
                sk_y1 = sk_y0 + (y_max - y_min)
                sk_x0 = int(hs - (xc - x_min))
                sk_x1 = sk_x0 + (x_max - x_min)
                sk = spatial_kernel[sk_y0:sk_y1, sk_x0:sk_x1]

                weights = np.empty_like(window)
                for wi in range(window.shape[0]):
                    for wj in range(window.shape[1]):
                        dv = window[wi, wj] - vc
                        weights[wi, wj] = sk[wi, wj] * np.exp(- (dv * dv) / (2 * hr * hr))

                total_weight = np.sum(weights)
                if total_weight < 1e-5:
                    break

                mean_x = 0.0
                mean_y = 0.0
                mean_v = 0.0
                for wi in range(window.shape[0]):
                    for wj in range(window.shape[1]):
                        px = x_min + wj
                        py = y_min + wi
                        w_ = weights[wi, wj]
                        mean_x += w_ * px
                        mean_y += w_ * py
                        mean_v += w_ * window[wi, wj]
                mean_x /= total_weight
                mean_y /= total_weight
                mean_v /= total_weight

                shift = np.sqrt((mean_x - xc)**2 + (mean_y - yc)**2 + (mean_v - vc)**2)
                xc, yc, vc = mean_x, mean_y, mean_v

                if shift < eps:
                    break

            output[i, j] = vc

    return np.clip(output, 0, 255).astype(np.uint8)

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

def BilateralFilterGrayscale(I, ss, sr, n_iter=1, n=3):
    """
    Applies a fast bilateral filter to a grayscale image using raised cosine approximation.

    Parameters:
    ----------
    I : np.ndarray
        Input grayscale image (uint8 or float32), values in range [0, 255] or [0.0, 1.0].
    ss : float
        Spatial standard deviation (sigma_s) of the Gaussian kernel.
    sr : float
        Range standard deviation (sigma_r) for intensity differences.
    n_iter : int, optional
        Number of iterations to apply the filter (default is 1).
    n : int, optional
        Degree of the raised cosine approximation (default is 3).

    Returns:
    -------
    np.ndarray
        Filtered grayscale image in uint8 format, values in range [0, 255].
    """
    I = I.astype(np.float32) / 255.0  # Normalize input image to [0,1]
    g = 1.0 / sr  # Inverse of range standard deviation
    sqrt_n = np.sqrt(n)

    h, w = I.shape

    binomial_coeffs = np.array([comb(n, i, exact=False) / (2 ** (2 * n - 2)) for i in range(n + 1)], dtype=np.float32)

    for _ in range(n_iter):
        num = np.zeros_like(I, dtype=np.float32)
        den = np.zeros_like(I, dtype=np.float32)

        for i in range(n + 1):
            v = g * (2 * i - n) * I / sqrt_n  # Argument of cosine and sine

            # Weight factor (binomial coefficient)
            b = binomial_coeffs[i]

            cos_v = np.cos(v)
            sin_v = np.sin(v)

            Hi0 = cos_v
            Hi1 = sin_v

            Gi0 = I * Hi0
            Gi1 = I * Hi1

            # Apply Gaussian filtering to both components
            Gir0 = gaussian_filter(Gi0, sigma=ss)
            Gir1 = gaussian_filter(Gi1, sigma=ss)
            Hir0 = gaussian_filter(Hi0, sigma=ss)
            Hir1 = gaussian_filter(Hi1, sigma=ss)

            # Accumulate numerator and denominator
            num += b * (Hi0 * Gir0 + Hi1 * Gir1)
            den += b * (Hi0 * Hir0 + Hi1 * Hir1)

        # Avoid division by zero
        I = np.divide(num, den, out=np.zeros_like(I), where=den != 0)

    return np.clip(I * 255, 0, 255).astype(np.uint8)

def BilateralFilterBGR(I_bgr, ss, sr, niter=1, n=3):
    """
    Applies the fast bilateral filter to a BGR color image using elevated cosine approximation.
    
    Parameters:
    - I_bgr: Input image in BGR format (numpy array of shape HxWx3, dtype=uint8).
    - ss: Spatial standard deviation (sigma for spatial Gaussian).
    - sr: Range standard deviation (sigma for intensity Gaussian).
    - niter: Number of iterations (default: 1).
    - n: Approximation parameter; higher values approximate the Gaussian more closely (recommended: 3-5).
    
    Returns:
    - Filtered image in BGR format (numpy array of dtype=uint8).
    """
    # Normalize the input to [0,1]
    I_bgr = I_bgr.astype(np.float32) / 255.0
    g = 1.0 / sr
    h, w, c = I_bgr.shape
    output = np.empty_like(I_bgr)

    for ch in range(3):  # Process each BGR channel independently
        I = I_bgr[..., ch]
        for _ in range(niter):
            num = np.zeros_like(I, dtype=np.float32)
            den = np.zeros_like(I, dtype=np.float32)

            for i in range(n + 1):
                v = g * (2 * i - n) * I / np.sqrt(n)
                b = comb(n, i, exact=False) / (2 ** (2 * n - 2))

                Hi = np.stack([np.cos(v), np.sin(v)], axis=-1)
                Gi = I[..., np.newaxis] * Hi
                Di = Hi * b

                Gir = np.stack([gaussian_filter(Gi[..., 0], sigma=ss),
                                gaussian_filter(Gi[..., 1], sigma=ss)], axis=-1)
                Hir = np.stack([gaussian_filter(Hi[..., 0], sigma=ss),
                                gaussian_filter(Hi[..., 1], sigma=ss)], axis=-1)

                num += Di[..., 0] * Gir[..., 0] + Di[..., 1] * Gir[..., 1]
                den += Di[..., 0] * Hir[..., 0] + Di[..., 1] * Hir[..., 1]

            I = np.divide(num, den, out=I, where=den != 0)

        output[..., ch] = I

    # Convert back to uint8
    return (np.clip(output, 0, 1) * 255).astype(np.uint8)

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
    img_smoothed = ConvolveSeparableOpt(img, gauss, gauss)

    return (img_smoothed * 255).astype(np.uint8)  # Convert back to uint8

def GaussianFilterBGR(img, sigma):
    """
    Applies a Gaussian filter to an BGR image.

    Parameters:
    - img: BGR image (numpy array).
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
        img_smoothed[:, :, i] = ConvolveSeparableOpt(img[:, :, i], gauss, gauss)

    # Convert the smoothed image back to uint8
    return (img_smoothed * 255).astype(np.uint8)

def AnisotropicDiffusion(img, num_iter=15, k=15, lamb=0.25, option=1):
    """
    Realiza difusión anisotrópica (Perona-Malik) en una imagen en escala de grises.

    Parámetros:
    - img: imagen de entrada (2D numpy array, escala de grises).
    - num_iter: número de iteraciones.
    - k: parámetro de sensibilidad a los bordes.
    - lambda_val: coeficiente de difusión (0 < lambda <= 0.25).
    - option: tipo de función conductiva (1 = exponencial, 2 = racional).

    Devuelve:
    - Imagen suavizada.
    """
    img = img.astype(np.float32)
    for _ in range(num_iter):
        # Gradientes en las 4 direcciones cardinales
        north = np.zeros_like(img)
        south = np.zeros_like(img)
        east = np.zeros_like(img)
        west = np.zeros_like(img)

        north[1:, :] = img[:-1, :] - img[1:, :]
        south[:-1, :] = img[1:, :] - img[:-1, :]
        east[:, :-1] = img[:, 1:] - img[:, :-1]
        west[:, 1:] = img[:, :-1] - img[:, 1:]

        # Función conductiva
        if option == 1:
            c_n = np.exp(-(north / k)**2)
            c_s = np.exp(-(south / k)**2)
            c_e = np.exp(-(east / k)**2)
            c_w = np.exp(-(west / k)**2)
        elif option == 2:
            c_n = 1.0 / (1.0 + (north / k)**2)
            c_s = 1.0 / (1.0 + (south / k)**2)
            c_e = 1.0 / (1.0 + (east / k)**2)
            c_w = 1.0 / (1.0 + (west / k)**2)
        else:
            raise ValueError("option debe ser 1 (exp) o 2 (racional)")

        # Actualización
        img += lamb * (
            c_n * north +
            c_s * south +
            c_e * east +
            c_w * west
        )
    return img.clip(0, 255).astype(np.uint8) if img.max() <= 255 else img.clip(0, 1)

# ---------------------------------
#                                   #
#   Threshold functions            #
#                                   #
# ---------------------------------

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

    return T_old

def OtsuThreshold(I):
    """Implementación del algoritmo de Otsu para encontrar el umbral óptimo."""
    # 1. Calcular el histograma
    hist = CalHistogram(I)
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

def flood_fill(image, x, y, visited):
    """ Propaga la conexión de bordes fuertes a bordes débiles (8 vecinos) """
    h, w = image.shape
    stack = [(x, y)]
    
    while stack:
        px, py = stack.pop()
        
        if visited[py, px]:  # Si ya fue visitado, saltar
            continue
        
        visited[py, px] = True
        image[py, px] = 255  # Marcar como borde fuerte
        
        # Revisar vecinos en 8 direcciones
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = px + dx, py + dy
                if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx] and image[ny, nx] != 0:
                    stack.append((nx, ny))

def HysteresisThreshold(image, T_low, T_high):
    """ Aplica umbralización por histéresis """
    # Detectar bordes débiles y fuertes
    strong_edges = (image >= T_high).astype(np.uint8) * 255
    weak_edges = ((image >= T_low) & (image < T_high)).astype(np.uint8) * 255
    
    # Imagen resultado inicializada en ceros
    result = np.zeros_like(image, dtype=np.uint8)
    visited = np.zeros_like(image, dtype=np.bool_)
    
    # Propagar bordes fuertes usando flood fill
    h, w = image.shape
    for y in range(h):
        for x in range(w):
            if strong_edges[y, x] == 255 and not visited[y, x]:
                flood_fill(weak_edges, x, y, visited)  # Expande bordes fuertes
    
    return weak_edges  # Retorna la imagen binaria con los bordes confirmados

# ---------------------------------#                                 
#   Fourier Transform functions    #
#                                  #
# ---------------------------------#

def FourierTransform2D(image):
    """
    Computes the 2D Fourier Transform of an image step by step
    using two 1D FFTs: first along rows, then along columns.
    
    Parameters:
        image: Input image (2D numpy array).
    
    Returns:
        2D Fourier Transform of the image.
    """
    # First, apply FFT along the rows (axis=1)
    fft_rows = np.fft.fft(image, axis=1)
    
    # Then, apply FFT along the columns (axis=0)
    fft2d = np.fft.fft(fft_rows, axis=0)
    
    return fft2d

def SlowFourier2D(image):
    """
    Computes the 2D Discrete Fourier Transform (DFT) of a grayscale image using the direct definition.

    This implementation follows the mathematical formula for the 2D DFT:

        F(u, v) = ∑∑ f(x, y) · exp(-j·2π·(ux/M + vy/N))

    where:
        - f(x, y) is the pixel value at coordinates (x, y),
        - M and N are the image dimensions,
        - u and v are the frequency domain coordinates.

    Parameters:
        image (2D numpy array): Grayscale input image.

    Returns:
        2D numpy array of complex numbers: The frequency-domain representation of the image.

    Notes:
        - This is a very slow implementation with time complexity O(N^4).
        - Intended for educational purposes only. For practical use, prefer numpy.fft.fft2().
    """
    M, N = image.shape
    output = np.zeros((M, N), dtype=complex)

    for u in range(M):
        for v in range(N):
            sum_real = 0.0
            sum_imag = 0.0
            for x in range(M):
                for y in range(N):
                    angle = -2 * np.pi * ((u * x / M) + (v * y / N))
                    e = np.exp(1j * angle)
                    sum_real += image[x, y] * e.real
                    sum_imag += image[x, y] * e.imag
            output[u, v] = sum_real + 1j * sum_imag

    return output

def ComputeSpectraFromDFT(dft):
    """
    Computes and returns the magnitude and phase spectra from a 2D Fourier Transform.

    This function takes a 2D Fourier Transform (complex-valued input) and computes:
    The magnitude spectrum, which is log-scaled and normalized for visualization.
    - The phase spectrum, normalized to the range [0, 255] for visualization.

    Parameters:
        dft : np.ndarray
            2D numpy array representing the Fourier Transform of an image (complex-valued input).

    Returns:
        magnitude : np.ndarray
            2D array (uint8) representing the normalized log-magnitude spectrum.

        phase : np.ndarray
            2D array (uint8) representing the normalized phase spectrum in the range [0, 255].

    Notes:
    -----
    - The input `dft` is assumed to be the result of a 2D Fourier Transform.
    - Zero-frequency components are shifted to the center using `np.fft.fftshift`.
    - The magnitude spectrum is log-scaled to enhance visibility of low-intensity frequencies.
    - The phase spectrum is normalized to the range [0, 255] for visualization purposes.
    """

    # Compute magnitude and apply log-scaling
    magnitude = np.abs(dft)
    magnitude = np.log1p(magnitude)  # log(1 + |F(u,v)|)
    magnitude = np.fft.fftshift(magnitude)
    magnitude = (magnitude / np.max(magnitude) * 255).astype(np.uint8)

    # Compute and normalize phase
    phase = np.angle(dft)
    phase = np.fft.fftshift(phase)
    phase = (phase + np.pi) / (2 * np.pi) * 255  # Normalize to [0, 255]
    phase = phase.astype(np.uint8)

    return magnitude, phase

def ComputeFourierSpectra(image):
    """
    Computes and returns the magnitude and phase spectra of a 2D Fourier Transform.

    This function takes a grayscale image as input, applies the 2D Fourier Transform,
    and returns both the magnitude and phase spectra. The magnitude spectrum is 
    log-scaled and normalized to enhance visibility, while the phase spectrum is also 
    normalized to the 0–255 range for visualization purposes.

    Parameters:
        image : np.ndarray
            2D numpy array representing a grayscale image (real-valued input).

    Returns:
        magnitude : np.ndarray 2D array (uint8) representing the normalized log-magnitude spectrum.

        phase : np.ndarray 2D array (uint8) representing the normalized phase spectrum in the range [0, 255].

    Notes:
    -----
    - The function assumes the input is a 2D image.
    - Zero-frequency components are shifted to the center using `np.fft.fftshift`.
    - The Fourier transform is computed using a custom implementation `FourierTransform2D`.
    """
    dft = FourierTransform2D(image)

    # Compute magnitude and apply log-scaling
    magnitude = np.abs(dft)
    magnitude = np.log1p(magnitude)  # log(1 + |F(u,v)|)
    magnitude = np.fft.fftshift(magnitude)
    magnitude = (magnitude / np.max(magnitude) * 255).astype(np.uint8)

    # Compute and normalize phase
    phase = np.angle(dft)
    phase = np.fft.fftshift(phase)
    phase = (phase + np.pi) / (2 * np.pi) * 255  # Normalize to [0, 255]
    phase = phase.astype(np.uint8)

    return magnitude, phase


# --------------------------------- #
#                                   #
#   Edge detection functions        #
#                                   #
# --------------------------------- #

def GaussianPyramid(image, levels=5, factor=2):
    """
    Applies Gaussian smoothing and downsampling to an image, constructing a Gaussian pyramid.
    Args:
        image (np.ndarray): Input image to be processed.
        levels (int, optional): Number of pyramid levels to generate. Default is 5.
        factor (int or float, optional): Downsampling factor between levels. Default is 2.
    Returns:
        list of np.ndarray: List containing the images at each level of the pyramid, starting with the original image.
    Notes:
        - Requires `lip.ConvolveSeparableOpt` for Gaussian smoothing and `zoom` for downsampling.
        - `gaussian_kernel_1d` must be defined and provided for the convolution.
    """
    gaussian_kernel_1d = np.array([0.25, 0.5, 0.25])
    pyramid = [image]
    current = image.copy()
    
    for i in range(1, levels):
        
        # Smooth the current image using the Gaussian kernel
        smoothed = ConvolveSeparableOpt(current, gaussian_kernel_1d, gaussian_kernel_1d)
        
        # Downsample the smoothed image
        # Using zoom from scipy to downsample the image by the specified factor
        # The order=1 corresponds to bilinear interpolation
        # This is a common choice for downsampling to avoid aliasing artifacts
        downsampled = zoom(smoothed, zoom=1/factor, order=1)
        pyramid.append(downsampled)
        current = downsampled

    return pyramid

def LaplacianGaussianPyramid(image, levels=5, factor=2, sigma=1.0):
    """
    Applies Gaussian smoothing and downsampling to an image, constructing a Gaussian pyramid.
    Args:
        image (np.ndarray): Input image to be processed.
        levels (int, optional): Number of pyramid levels to generate. Default is 5.
        factor (int or float, optional): Downsampling factor between levels. Default is 2.
    Returns:
        list of np.ndarray: List containing the images at each level of the pyramid, starting with the original image.
    Notes:
        - Requires `lip.ConvolveSeparableOpt` for Gaussian smoothing and `zoom` for downsampling.
        - `gaussian_kernel_1d` must be defined and provided for the convolution.
    """
    pyramid = []
    current = image.copy()
    
    for i in range(0, levels):
        
        # Smooth the current image using the Gaussian kernel
        smoothed = ComputeLaplacianGaussian(current, sigma, sigma)
        
        # Downsample the smoothed image
        # Using zoom from scipy to downsample the image by the specified factor
        # The order=1 corresponds to bilinear interpolation
        # This is a common choice for downsampling to avoid aliasing artifacts
        downsampled = zoom(smoothed, zoom=1/factor, order=1)
        pyramid.append(downsampled)
        current = downsampled

    return pyramid

def NonMaxSuppression(Gmag, Gphase):
    """
    Applies non-maximum suppression to the gradient magnitude image.

    Parameters:
        Gmag (numpy.ndarray): Gradient magnitude image.
        Gphase (numpy.ndarray): Gradient phase image (in radians).

    Returns:
        numpy.ndarray: Gradient magnitude image with non-local maxima suppressed.
    """
    # Convert phase angles to degrees for easier comparison
    Gphase = np.rad2deg(Gphase) % 180  # Normalize phase to [0, 180)

    # Initialize the output image
    Glocalmax = np.zeros_like(Gmag, dtype=np.float32)

    # Get image dimensions
    rows, cols = Gmag.shape

    for x in range(1, rows - 1):
        for y in range(1, cols - 1):
            # Get the gradient direction at the current pixel
            u = Gphase[x, y]

            # Determine the neighbors based on the gradient direction
            if (0 <= u < 22.5) or (157.5 <= u <= 180):  # Horizontal edge
                neigh1 = Gmag[x, y - 1]
                neigh2 = Gmag[x, y + 1]
            elif 22.5 <= u < 67.5:  # Diagonal edge (top-left to bottom-right)
                neigh1 = Gmag[x - 1, y - 1]
                neigh2 = Gmag[x + 1, y + 1]
            elif 67.5 <= u < 112.5:  # Vertical edge
                neigh1 = Gmag[x - 1, y]
                neigh2 = Gmag[x + 1, y]
            elif 112.5 <= u < 157.5:  # Diagonal edge (top-right to bottom-left)
                neigh1 = Gmag[x - 1, y + 1]
                neigh2 = Gmag[x + 1, y - 1]

            # Suppress non-local maxima
            if Gmag[x, y] >= neigh1 and Gmag[x, y] >= neigh2:
                Glocalmax[x, y] = Gmag[x, y]
            else:
                Glocalmax[x, y] = 0

    return Glocalmax

def NonMaximumSuppressionSubpixel(Gx: np.ndarray, Gy: np.ndarray, Gmag: np.ndarray) -> np.ndarray:
    rows, cols = Gmag.shape
    output = np.zeros_like(Gmag, dtype=np.float32)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            gx, gy = Gx[i, j], Gy[i, j]
            mag = Gmag[i, j]
            if gx == 0 and gy == 0:
                continue
            norm = np.hypot(gx, gy)
            dx, dy = gx / norm, gy / norm

            def interp(y, x):
                x0, y0 = int(x), int(y)
                x1 = min(x0 + 1, cols - 1)
                y1 = min(y0 + 1, rows - 1)
                a, b = x - x0, y - y0
                return (
                    Gmag[y0, x0] * (1 - a) * (1 - b) +
                    Gmag[y0, x1] * a * (1 - b) +
                    Gmag[y1, x0] * (1 - a) * b +
                    Gmag[y1, x1] * a * b
                )

            mag1 = interp(i + dy, j + dx)
            mag2 = interp(i - dy, j - dx)

            if mag >= mag1 and mag >= mag2:
                output[i, j] = mag
    return output

def HysteresisThresholdFIFO(image, T_high, T_low):
    h, w = image.shape
    strong = (image >= T_high)
    weak = (image >= T_low) & ~strong
    result = np.zeros_like(image, dtype=np.uint8)
    queue = deque()

    for y in range(h):
        for x in range(w):
            if strong[y, x]:
                result[y, x] = 255
                queue.append((x, y))

    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),         (0, 1),
                  (1, -1), (1, 0), (1, 1)]

    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                if weak[ny, nx] and result[ny, nx] == 0:
                    result[ny, nx] = 255
                    queue.append((nx, ny))
    return result

def CannyLikeDetector(image: np.ndarray, sigma=1.0, tlow=0.1, thigh=0.3) -> np.ndarray:
    """
    Performs edge detection on a single-channel (grayscale) image using a Canny-like algorithm.
    This function computes image gradients, applies non-maximum suppression, normalizes the result,
    and performs hysteresis thresholding to detect edges. The input image must be a 2D NumPy array
    representing a grayscale image (i.e., with shape [height, width] and a single channel).
    Parameters:
        image (np.ndarray): Input grayscale image as a 2D NumPy array (single channel).
        sigma (float, optional): Standard deviation for Gaussian smoothing in gradient computation. Default is 1.0.
        tlow (float, optional): Lower threshold for hysteresis, as a fraction of 255. Default is 0.1.
        thigh (float, optional): Higher threshold for hysteresis, as a fraction of 255. Default is 0.3.
    Returns:
        np.ndarray: Binary edge map as a 2D NumPy array of the same shape as the input image.
    """
    
    Gx, Gy, Gmag, Gphase = ComputeImageGradient(image, sigma_s=sigma, sigma_d=sigma)
    suppressed = NonMaximumSuppressionSubpixel(Gx, Gy, Gmag)

    # Normalización post-supresión
    norm_suppressed = np.clip((suppressed / suppressed.max()) * 255.0, 0, 255).astype(np.float32)

    # Umbrales escalados
    T_high = thigh * 255
    T_low = tlow * 255

    edges = HysteresisThresholdFIFO(norm_suppressed, T_high, T_low)
    return edges
