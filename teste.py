from Linha import Linha


linha = Linha()

linha.append(1,1,100,100)
linha.append(10,10,160,160)
linha.append(165,165,500,500)

print(linha.getLinha(1))

print(linha.contem(0,0,162,162,limite=5))

print(linha.getLinhaFinal())