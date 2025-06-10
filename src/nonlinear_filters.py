import cv2
import matplotlib.pyplot as plt
import lip

def main():
    
    img_g = cv2.imread("img_data/lena_salt.jpg", cv2.IMREAD_GRAYSCALE)    
    img_c = cv2.imread("img_data/lena_salt.jpg")

    img_gb = lip.GaussianFilterGrayscale(img_g, 1.5)
    img_cb = lip.GaussianFilterBGR(img_c, 1.5)
    lip.show_images_together([img_g, img_gb, img_c, img_cb])
    
    img_gf = lip.MedianFilterGrayscale(img_g, 3)
    img_cf = lip.MedianFilterBGR(img_c, 3)
    lip.show_images_together([img_g, img_gf, img_c, img_cf],)
    
    img_g = cv2.imread("img_data/desert1.jpg", cv2.IMREAD_GRAYSCALE)    
    img_c = cv2.imread("img_data/desert1.jpg")    
    
    img_g = cv2.resize(img_g, (0, 0), fx=0.7, fy=0.7)
    img_c = cv2.resize(img_c, (0, 0), fx=0.7, fy=0.7)

    img_gbi = lip.BilateralFilterGrayscale(img_g, 1.2, 0.7, 8, 5)
    img_cbi = lip.BilateralFilterBGR(img_c, 1.2, 0.7, 8, 5)
    
    lip.show_images_together([img_g, img_gbi, img_c, img_cbi],)    
    
    img_g = cv2.imread("img_data/lienzo.jpg", cv2.IMREAD_GRAYSCALE)    
    img_c = cv2.imread("img_data/lienzo.jpg")

    img_g = cv2.resize(img_g, (0, 0), fx=0.8, fy=0.8)
    img_c = cv2.resize(img_c, (0, 0), fx=0.8, fy=0.8)
    
    img_gms = lip.MeanShiftFilterGrayscale(img_g, 10, 5, 5)
    img_cms = lip.MeanShiftFilterBGR(img_c, 10, 5, 5)
    
    lip.show_images_together([img_g, img_gms, img_c, img_cms],)
    
    #this takes a long time to run, so it is commented out
    '''
    img_g = cv2.imread("img_data/naturalezaM.jpg", cv2.IMREAD_GRAYSCALE)    
    img_c = cv2.imread("img_data/naturalezaM.jpg")

    img_g = cv2.resize(img_g, (0, 0), fx=1.6, fy=1.6)
    img_c = cv2.resize(img_c, (0, 0), fx=1.6, fy=1.6)
    
    img_gms = lip.MeanShiftFilterGrayscale(img_g, 32, 15.5, 7)
    img_cms = lip.MeanShiftFilterBGR(img_c, 32, 15.5, 7)
    
    lip.show_images_together([img_g, img_gms, img_c, img_cms],)
    '''

if __name__ == "__main__":
    main()