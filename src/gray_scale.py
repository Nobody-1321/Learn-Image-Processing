import cv2 as cv
import lip  # This should include your custom BgrToGray implementation

def main():
    # Load the input image in BGR format (default behavior of OpenCV)
    img1 = cv.imread("img_data/astro.jpg")    
    img1 = cv.resize(img1, (600, 600), interpolation=cv.INTER_AREA)

    # Convert the image to grayscale using OpenCV's built-in function
    img2_cv = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

    # Convert the image to grayscale using custom function from 'lip' module
    img2_lip = lip.BgrToGray(img1)

    # Show both grayscale images side-by-side for comparison
    lip.show_images_together([img2_cv, img2_lip], ["opencv", "lip"])

if __name__ == '__main__':
    main()
