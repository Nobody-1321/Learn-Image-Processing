import ena
import numpy as np
import cv2 as cv

def main():

    #img = cv.imread("img_data/salt_papper.png")
    img = cv.imread("img_data/astro.jpg")
    
    
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    img = cv.resize(img, (800, 800))

    #img_noise = ena.AddGaussianNoise(img, 50.0)
    #img_noise = ena.AddSaltAndPepperNoise(img, 0.02, 0.08)
    #img_noise = ena.AddPoissonNoise(img)
    #img_noise = ena.AddUniformNoise(img, 10, 100)
    img_noise = ena.AddSpeckleNoise(img, 0.1)


    ena.show_two_images_side_by_side(img, img_noise, "Original vs Noise", "horizontal")
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()