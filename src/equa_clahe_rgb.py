import cv2 as cv 
import lip

#img = cv.imread("img_data/Moises.jpg")
#img = cv.imread("img_data/white_monta.jpg")
#img = cv.imread("img_data/ciervo.jpg")
#img = cv.imread("img_data/sun.jpg")
#img = cv.imread("img_data/caballo.webp")
img = cv.imread("img_data/input.jpg")
#img = cv.imread("img_data/monta.jpg")
#img = cv.imread("img_data/noct.jpeg")
#img = cv.imread("img_data/Rose.jpg")
#img = cv.imread("img_data/compressed.jpeg")
#img = cv.imread("img_data/sky.webp")
#img = cv.imread("img_data/blue_dark.jpg")
#img = cv.imread("img_data/M4.jpg")
#img = cv.imread('img_data/astro.jpg')

img = cv.resize(img, (720, 480), interpolation=cv.INTER_AREA)


img_equa =  lip.HistogramEqualization(img)
img_clahe = lip.HistogramEqualizationClaheRGB(img, clip_limit=20, grid_size=(9,9))

#lip.plot_three_images_and_histograms(img, img_equa, img_clahe, "Original", "Histogram Equalization", "CLAHE")
lip.show_images_together([img, img_clahe],[".", "."])
