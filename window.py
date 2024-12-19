import cv2
import numpy as np

imagen = cv2.imread('img/YU.jpg')

imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
imagen = cv2.resize(imagen, (800, 600), interpolation=cv2.INTER_LINEAR)

'''
imagen = imagen.astype('float16')
imagen += 50
imagen = np.clip(imagen, 0, 255)
imagen = imagen.astype('uint8')
'''
imagen = cv2.equalizeHist(imagen)


cv2.imwrite('YU.jpg', imagen)

cv2.destroyAllWindows()



