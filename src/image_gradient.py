import cv2
import matplotlib.pyplot as plt
import lip

def main():

    #img = cv2.imread("img_data/astro.jpg", cv2.IMREAD_GRAYSCALE)
    img = cv2.imread("img_data/caballo.webp", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (500, 400))

    Gradient_x, Gradient_y, Gradient_mag, Gradient_phase = lip.ComputeImageGradient(img, 0.5, 1.0)

    lip.show_images_together([Gradient_x, Gradient_y, Gradient_mag, Gradient_phase], ["A", "B", "C", "D"])
    # LOG 
    img = cv2.resize(img, (800, 600))
    GradientLog = lip.ComputeLaplacianGaussian(img, 0.5, 0.7)   
    cv2.imshow("LOG", GradientLog)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()