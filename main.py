import cv2 as cv
import numpy as np
from Linha import Linha

imagens= ['imagem1.png','imagem2.png','imagem3.png','imagem4.png','imagem5.png','imagem6.png']

def kernel(r):
    return np.fromfunction(lambda x, y: ((x-r)**2 + (y-r)**2 <= r**2)*1, (2*r+1, 2*r+1), dtype=int).astype(np.uint8)


img = cv.imread("img/" + imagens[4])
#img = cv.resize(img,(600,400))
original = img.copy()



#img = cv.medianBlur(img,3)
img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
print(f'Tamanho imagem: {img.shape[0]}x{img.shape[1]}')

ret,img = cv.threshold(img,220,255,cv.THRESH_BINARY)

filter = kernel(16)
img = cv.dilate(img, filter, iterations=1)
img = cv.erode(img, filter, iterations=1)

filter = kernel(8)
img = cv.dilate(img, filter, iterations=1)
img = cv.erode(img, filter, iterations=1)


result = np.zeros(img.shape, np.uint8)

            

edges = cv.Canny(img,50,200)
'''kernel = np.ones((5,5), np.uint8)
edges = cv.dilate(edges, kernel, iterations=1)
edges = cv.erode(edges, kernel, iterations=1)'''
#lines = cv.HoughLinesP(edges,1,np.pi/180,60,minLineLength=150,maxLineGap=50)
lines = cv.HoughLines(edges,1,np.pi/180,110)

print(f'Linhas encontradas = {len(lines)}')

'''linhasFinal = []
limite = 0

for i in range(len(lines)):
    x1,y1,x2,y2 = lines[i][0]
    repetida = False
    for linha in linhasFinal:s
        if linha.contem(x1,y1,x2,y2,limite = limite):
            repetida = True
            linha.append(x1,y1,x2,y2)

    if not repetida:
        novaLinha = Linha(x1,y1,x2,y2)
        linhasFinal.append(novaLinha)

for linha in linhasFinal:
    cv.line(result,linha.getPontoInicial(),linha.getPontoFinal(),(255,0,0),1) 
print(f'Linhas desenhadas = {len(linhasFinal)}') '''     
 
# Filtrando
finalLines = []
thresholdRho = 30 # Porcentagem
thresholdTheta = 0.1 # Porcentagem

for i in range(len(lines)):
    rho = lines[i][0][0]
    theta = lines[i][0][1]
    match = False

    for j in range(len(finalLines)):
        rhoRef, thetaRef = finalLines[j]
        if rho >= rhoRef-thresholdRho and rho <= rhoRef+thresholdRho and theta >= thetaRef-thresholdTheta and theta <= thetaRef+thresholdTheta:
            finalLines.append(((rho+rhoRef)/2,(theta+thetaRef)/2))
            finalLines.pop(j)
            match = True
            

    if not match:
        finalLines.append((rho,theta))
    
print(f'Linhas filtradas = {len(finalLines)}')

for i in range(len(finalLines)):
    '''x1,y1,x2,y2 = lines[i][0]
    cv.line(result,(x1,y1),(x2,y2),(255,0,0),1) '''

    rho, theta = finalLines[i]
    a = np.cos(theta)
    b = np.sin(theta)
    print(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
    pt2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
    if theta > (np.pi/2):
        # Linhas "descendo"
        colorBGR = (0,0,255) 
        colorGray = 255
    elif theta > (np.pi/2)*0.1:
        # Linhas "subindo"
        colorBGR = (255,0,0)
        colorGray = 255
    else: 
        # Linhas quase em pe (traves do gol)
        colorBGR = (0,255,0)
        colorGray = 0
    cv.line(original, pt1, pt2, colorBGR, 3)
    cv.line(result, pt1, pt2, colorGray, 3)



# Interseccoes

#for i in range(len(finalLines)):

    

cv.imshow("original",original)
#cv.imshow("mascara",img)
#cv.imshow("edges",edges)
cv.imshow("res",result)


#cv.imshow('imagem',img)

#cv.imwrite('results/original with lines (24-05.png)',original)
#cv.imwrite('results/lines (24-05).png',result)


cv.waitKey(0)
cv.destroyAllWindows()