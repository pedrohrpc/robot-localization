import cv2 as cv
import numpy as np
from Interseccao import Interseccao

imagens= ['imagem1.png','imagem2.png','imagem3.png','imagem4.png','imagem5.png','imagem6.png']

def kernel(r):
    return np.fromfunction(lambda x, y: ((x-r)**2 + (y-r)**2 <= r**2)*1, (2*r+1, 2*r+1), dtype=int).astype(np.uint8)


img = cv.imread("img/" + imagens[1])

# Cortando a imagem para ficar quadrada
if img.shape[0] < img.shape[1]:
    aux =   img.shape[1] - img.shape[0]
    img = img[0:img.shape[0], int(aux/2):img.shape[1]-int(aux/2)]
else: 
    aux =   img.shape[0] - img.shape[1]
    img = img[int(aux/2):img.shape[0]-int(aux/2), 0:img.shape[1]]

razao = img.shape[0]
img = cv.resize(img,(416,416))
razao = razao/img.shape[0]
print(f'Razao de tamanos: {razao}')

original = img.copy()
imagemFinal = img.copy()

#img = cv.medianBlur(img,3)
img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
print(f'Tamanho imagem: {img.shape[0]}x{img.shape[1]}')

ret,mask = cv.threshold(img,220,255,cv.THRESH_BINARY)

filter = kernel(int(16/razao))
mask = cv.dilate(mask, filter, iterations=1)
mask = cv.erode(mask, filter, iterations=1)

filter = kernel(int(10/razao))
mask = cv.dilate(mask, filter, iterations=1)
mask = cv.erode(mask, filter, iterations=1)

filter = kernel(int(16/razao))
verifyIntersec = cv.dilate(mask, filter, iterations=1)


result = np.zeros(mask.shape, np.uint8)


edges = cv.Canny(mask,50,200)
lines = cv.HoughLines(edges,1,np.pi/180,int(80/razao))

print(f'Linhas encontradas = {len(lines)}')

 
# Filtrando
finalLines = []
thresholdRho = 30
thresholdTheta = 0.2

for i in range(len(lines)):
    rho = lines[i][0][0]
    theta = lines[i][0][1]
    match = False

    for j in range(len(finalLines)):
        rhoRef, thetaRef = finalLines[j]
        if rho >= rhoRef-thresholdRho and rho <= rhoRef+thresholdRho and theta >= thetaRef-thresholdTheta and theta <= thetaRef+thresholdTheta:
            finalLines.append((rhoRef,thetaRef))
            finalLines.pop(j)
            match = True
            

    if not match:
        finalLines.append((rho,theta))
    
print(f'Linhas filtradas = {len(finalLines)}')

colorRed = (0,0,255) 
colorBlue = (255,0,0) 
colorGray = 255

for i in range(len(finalLines)):

    rho, theta = finalLines[i]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
    pt2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
    '''if theta > (np.pi/2):
        # Linhas "descendo"
        colorBGR = (0,0,255) 
        colorGray = 255
    else:
        # Linhas "subindo"
        colorBGR = (255,0,0)
        colorGray = 255'''
    
    cv.line(imagemFinal, pt1, pt2, colorRed, 3)
    cv.line(result, pt1, pt2, colorGray, 3)


# Interseccoes
# y = (-cos(theta)/sin(theta) * x) + rho/sin(theta)
# a = -cos(theta)/sin(theta)
# b = rho/sin(theta)
# y = a * x + b
# Interseccao => (x1,y1) = (x2,y2)
# Interseccao em x, x = (rho2/sin(theta2) - rho1/sin(theta1)) / (-cos(theta1)/sin(theta1) + cos(theta2)/sin(theta2))
# Interseccao em x, x = (b2 - b1) / (a1 - a2)
# Para interseccao estar na imagem, 0 <= x <= xmax

# Encontrando todas as interseccoes
interseccoes = []
for i in range(len(finalLines)-1,0,-1):
    rho1, theta1 = finalLines[i]
    a1 = -np.cos(theta1)/np.sin(theta1)
    b1 = rho1/np.sin(theta1)

    for j in range(len(finalLines)):
        rho2, theta2 = finalLines[j]
        if rho1 != rho2 and theta1 != theta2:
            '''a2 = -np.cos(theta2)/np.sin(theta2)
            b2 = rho1/np.sin(theta2)
            x = int((b2 - b1) / (a1 - a2))
            y = int(a1 * x + b1)'''
            A = np.array([[np.cos(theta1), np.sin(theta1)],[np.cos(theta2), np.sin(theta2)]])
            b = np.array([[rho1], [rho2]])
            x, y = np.linalg.solve(A, b)
            x, y = int(np.round(x)), int(np.round(y))
            
            if x >=0 and x <= original.shape[0] and y >=0 and y <= original.shape[1]:
                interseccao = Interseccao(x,y,(rho1,theta1),(rho2,theta2))
                interseccoes.append(interseccao)
    
    finalLines.pop(i)

# Filtrando e classfiicando interseccoes
interseccoes = list(set(interseccoes))
interseccoesFinais = []
for interseccao in interseccoes:
    p1, p2, p3, p4 = interseccao.vizinhos(15) #p1 e p3 pertencem a uma reta, p2 e p4 a outra
    p1Pertence = verifyIntersec[p1[0]][p1[1]] == 255
    p2Pertence = verifyIntersec[p2[0]][p2[1]] == 255
    p3Pertence = verifyIntersec[p3[0]][p3[1]] == 255
    p4Pertence = verifyIntersec[p4[0]][p4[1]] == 255
    if (p1Pertence or p3Pertence) and (p2Pertence or p4Pertence): #Para ser uma interseccao, precisa de no minimo um ponto em cada reta
        if p1Pertence and p3Pertence and p2Pertence and p4Pertence:
            interseccao.classificar(4)
        elif (p1Pertence and p3Pertence) or (p2Pertence and p4Pertence):
            interseccao.classificar(3)
        else:
            interseccao.classificar(2)
        interseccoesFinais.append(interseccao)

print(len(interseccoesFinais))
colors = [(0,0,0),(0,75,0),(0,150,0),(0,255,0)]

for interseccao in interseccoesFinais:
    print(interseccao.x,interseccao.y)
    cv.circle(imagemFinal,(interseccao.x,interseccao.y),3,colorBlue,3)


'''p1, p2, p3, p4 = interseccao.vizinhos(20)
cv.circle(imagemFinal,p1,2,(188,0,157),2)
cv.circle(imagemFinal,p3,2,(104,0,87),2)
cv.circle(imagemFinal,p2,2,(0,204,255),2)
cv.circle(imagemFinal,p4,2,(0,93,117),2)

cv.circle(verifyIntersec,p1,2,255,2)
cv.circle(verifyIntersec,p2,2,255,2)
cv.circle(verifyIntersec,p3,2,255,2)
cv.circle(verifyIntersec,p4,2,255,2)'''



camadas = np.concatenate((mask, edges), axis=1)
camadas = np.concatenate((camadas, np.concatenate((result, verifyIntersec), axis=1)), axis=0)
camadas = cv.resize(camadas, (result.shape[1],result.shape[0]))

coloridas = np.concatenate((original,imagemFinal), axis=1)

cv.imshow("original",imagemFinal)
cv.imshow("camadas",camadas)

cv.imshow('imagem',verifyIntersec)

#cv.imwrite('results/original with lines (24-05.png)',original)
#cv.imwrite('results/lines (24-05).png',result)

cv.waitKey(0)
cv.destroyAllWindows()