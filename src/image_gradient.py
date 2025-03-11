import cv2
import matplotlib.pyplot as plt
import lip

def main():

    img = cv2.imread("img_data/astro.jpg", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (400, 400))

    Gradient_x, Gradient_y, Gradient_mag, Gradient_phase = lip.ComputeImageGradient(img, 3.5)

    lip.show_images_together([Gradient_x, Gradient_y, Gradient_mag, Gradient_phase], ["Gradient X", "Gradient Y", "Magnitude", "Phase"])
    
if __name__ == "__main__":
    main()