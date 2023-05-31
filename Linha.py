
class Linha:
    minX = None
    minY = None
    maxX = None
    maxY = None
    linhasSimilares = []

    x1 = []
    y1 = []
    x2 = []
    y2 = []

    def __init__(self, x1,y1,x2,y2):
        self.append(x1, y1, x2, y2)

    def contem(self, x1, y1, x2, y2, limite):
        primeiroP = False
        ultimpP = False
        for i in range(len(self.x1)):
            if x1 < self.x1[i]+limite and x1 > self.x1[i]-limite and y1 < self.y1[i]+limite and y1 > self.y1[i]-limite:
                primeiroP = True
            
            if x2 < self.x2[i]+limite and x2 > self.x2[i]-limite and y2 < self.y2[i]+limite and y2 > self.y2[i]-limite:
                ultimpP = True
            

            

        return (primeiroP and ultimpP)

    def append(self, x1, y1, x2, y2):
        self.x1.append(x1)
        self.y1.append(y1)
        self.x2.append(x2)
        self.y2.append(y2)

        self.update()


    def update(self):
        self.minX = min(self.x1)
        self.minY = min(self.y1)
        self.maxX = max(self.x2)
        self.maxY = max(self.y2)

    def getLinha(self, indice):
        return (self.x1[indice],self.y1[indice],self.x2[indice],self.y2[indice])

    def getLinhaFinal(self):
        return (self.minX,self.minY,self.maxX,self.maxY)
        
    def getPontoInicial(self):
        return (self.minX,self.minY)

    def getPontoFinal(self):
        return (self.maxX,self.maxY)