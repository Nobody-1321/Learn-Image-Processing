import cv2
import matplotlib.pyplot as plt
import ena

def main():

    img = cv2.imread("img_data/salt_papper.png", cv2.IMREAD_GRAYSCALE)
    
    img = cv2.resize(img, (400, 400))

    img_smoothed = ena.MedianFilter(img, 5)

    ena.compare_histograms(img, img_smoothed, "Original", "Smoothed")


    cv2.imshow("Original Image", img)
    cv2.imshow("Smoothed Image", img_smoothed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
if __name__ == "__main__":
    main()