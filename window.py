import cv2


imagen = cv2.imread('img/Y4.jpg')



cv2.namedWindow('Img', cv2.WINDOW_NORMAL)

cv2.resizeWindow('Img', 800, 600)  # 800x600 píxeles

cv2.imshow('Img', imagen)

cv2.waitKey(0)
cv2.destroyAllWindows()
