from math import ceil, pow, sqrt
import numpy as np
import openml
import pandas as pd
from openml.datasets import edit_dataset, fork_dataset, get_dataset

#Estabelecendo porcentagem de treino
#-----------------------------------------------------------------------

trainPercentage = 0.6

if trainPercentage > 1:
    print("PORCENTAGEM SÓ VAI ATÉ UM, SEU BURRO!!!!!!!!")
    exit

# Adquirindo e processando o dataset
#-----------------------------------------------------------------------
dataset = openml.datasets.get_dataset(50)

X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="array", target=dataset.default_target_attribute)

df = pd.DataFrame(X, columns=attribute_names)
df['class'] = y

treino = df.sample(frac=trainPercentage, replace=False, axis=0)

print(treino['class'].value_counts())
exit()

#Distribuição igualitária dos resultados
#-----------------------------------------------------------------------
indice = ceil(len(X)*trainPercentage)

unique, counts = np.unique(y, return_counts=True)

indice0 = ceil(counts[1]*trainPercentage) #% de uns
indice1 = ceil(counts[0]*trainPercentage) #% de zeros

Xtreino = X[:indice0].tolist() + X[counts[1]:indice1].tolist()
Ytreino = y[:indice0].tolist() + y[counts[1]:indice1].tolist()
Xteste = X[indice0:counts[1]].tolist() + X[indice1:].tolist()
Yvalida = y[indice0:counts[1]].tolist() + y[indice1:].tolist()

#Knn Classificador
#-----------------------------------------------------------------------

k = 5

if k%2 == 0:
    print("WARNING: Para evitar empates, um valor ímpar é sempre recomendado!")

yteste = []

#Para cada valor de teste, existe uma lista de distâncias
for teste in Xteste:
    cont = 0
    listaDistancia = []
    #calcula as distâncias euclidianas de todas as amostras de treino
    for treino in Xtreino:
        dist = 0
        for i in range(0, len(treino)):
            dist = dist + pow(teste[i] - treino[i], 2)
        listaDistancia.append((sqrt(dist),cont))
        cont = cont + 1
    #organiza de acordo com a menor distância
    listaDistancia.sort(key=lambda tup: tup[0])
    #pega os k menores valores
    aux1 = 0 #0
    aux2 = 0 #1
    #(distancia, indice)
    for menor in range(0, k):
        tupla = listaDistancia[menor]
        if Ytreino[tupla[1]] == 0:
            aux1 = aux1 + 1
        elif Ytreino[tupla[1]] == 1:
            aux2 = aux2 + 1
    if aux1 > aux2:
        yteste.append(0)
    elif aux2 > aux1:
        yteste.append(1)
    else:
        print("Seu K está errado, caramba!")

#Cálculo das Métricas de acurácia e matriz de confusão
#-----------------------------------------------------------------------

TP = 0
TN = 0
FP = 0
FN = 0
controle = 0

# 0 == positivo
# 1 == negativo
for result in Yvalida:
    if result == 0:
        if result == yteste[controle]:
            TP = TP + 1
        else:
            FN = FN + 1
    else:
        if result == yteste[controle]:
            TN = TN + 1
        else:
            FP = FP + 1
    controle = controle + 1

ACC = (TP + TN)/(TP+TN+FP+FN)

print("Usando " + str(trainPercentage*100) + "% das amostras para treino ")
print(str(ACC*100) + "% de acurácia")