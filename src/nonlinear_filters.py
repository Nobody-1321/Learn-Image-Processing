import cv2
import matplotlib.pyplot as plt
import lip

def main():
    
    '''
    img_g = cv2.imread("img_data/lena_salt.jpg", cv2.IMREAD_GRAYSCALE)    
    img_c = cv2.imread("img_data/lena_salt.jpg")
    
    img_gb = lip.GaussianFilterGrayscale(img_g, 1.5)
    img_cb = lip.GaussianFilterBGR(img_c, 1.5)
    lip.show_images_together([img_g, img_gb, img_c, img_cb])
    
    img_gf = lip.MedianFilterGrayscale(img_g, 3)
    img_cf = lip.MedianFilterBGR(img_c, 3)
    lip.show_images_together([img_g, img_gf, img_c, img_cf],)
    '''
    
    #img_g = cv2.imread("img_data/desert1.jpg", cv2.IMREAD_GRAYSCALE)    
    #img_c = cv2.imread("img_data/desert1.jpg")
    
    img_g = cv2.imread("img_data/greenhead.webp", cv2.IMREAD_GRAYSCALE)    
    img_c = cv2.imread("img_data/greenhead.webp")    
    
    #img_g = cv2.resize(img_g, (0, 0), fx=0.5, fy=0.5)
    #img_c = cv2.resize(img_c, (0, 0), fx=0.5, fy=0.5)

    img_gbi = lip.BilateralFilterGrayscale(img_g, 0.8, 8.8, 5, 3)
    img_cbi = lip.BilateralFilterBGR(img_c, 1.5, 3.8, 5, 3)

    cv2.imshow("Bilateral Filter Grayscale", img_gbi)
    cv2.imshow("Bilateral Filter BGR", img_cbi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    '''
    img_cf1 = lip.MedianFilterBGR(img_c, 3)
    img_cf2 = lip.MedianFilterBGR(img_c, 5)
    img_cf3 = lip.MedianFilterBGR(img_c, 9)
    lip.show_images_together([img_c, img_cf1, img_cf2, img_cf3],
                             titles=["Original", "w3", "w5", "w7"])
    '''
    
if __name__ == "__main__":
    main()