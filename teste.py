import cv2 as cv
import numpy as np

img = np.zeros((1000,1000))
theta = np.pi/12
rho = 500

for i in range(3):
    thetaX = theta*i
    a = np.cos(thetaX)
    b = np.sin(thetaX)
    x0 = int(a * rho)
    y0 = int(b * rho)
    cv.circle(img,(x0,y0),5,255,5)

cv.imshow("camadas",img)


cv.waitKey(0)
cv.destroyAllWindows()