import cv2 as cv 
import lip
#img = cv.imread("img_data/Moises.jpg")
#img = cv.imread("img_data/white_monta.jpg")
#img = cv.imread("img_data/ciervo.jpg")
#img = cv.imread("img_data/noct.jpeg")
#img = cv.imread("img_data/Rose.jpg")
#img = cv.imread("img_data/compressed.jpeg")
#img = cv.imread("img_data/sky.webp")
#img = cv.imread("img_data/blue_dark.jpg")
img = cv.imread("img_data/M4.jpg")

#img = cv.imread('img_data/astro.jpg')
img = cv.resize(img, (600, 600), interpolation=cv.INTER_AREA)

img_equa = lip.HistogramEqualizationClaheRGB(img, clip_limit=5, grid_size=(13,13))

cv.imshow('Original', img)
cv.imshow('Equalized', img_equa)
cv.waitKey(0)
cv.destroyAllWindows()
