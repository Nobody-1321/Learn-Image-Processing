import lip
import numpy as np
import cv2 as cv

def main():

    #img = cv.imread("img_data/caballo.webp")
    #img = cv.imread("img_data/astro.jpg")
    img = cv.imread("img_data/paper.png")
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, (400, 400))

    # Convert to float32 for convolution
    img = np.float32(img)

    # Crear el kernel gaussiano
    gauss = lip.CreateGaussianKernel(1)
    # Aplicar la convolución separable
    img_smoothed = lip.ConvolveSeparable(img, gauss, gauss)
    
    #img_smoothed = ena.ConvolveBox(img, 20)
    
    print(img_smoothed)
    
    # Normalizar la imagen resultante para que esté en el rango [0, 255]
    #img_smoothed = np.clip(img_smoothed, 0, 255).astype(np.uint8)
    
    img_smoothed = cv.normalize(img_smoothed, None, 0, 255, cv.NORM_MINMAX)
    img_smoothed = np.uint8(img_smoothed)


    # Mostrar imágenes
    cv.imshow("Original", img.astype(np.uint8))
    cv.imshow("Smoothed Image", img_smoothed)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()