import cv2 as cv
import lip

def main():
    
    img1 = cv.imread("img_data/astro.jpg")    
    img1 = cv.resize(img1, (600, 600), interpolation=cv.INTER_AREA)
    
    img2_cv = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2_lip = lip.BgrToGray(img1)

    lip.show_images_together([img2_cv, img2_lip],["opencv", "lip"])



if __name__ == '__main__':
    main()