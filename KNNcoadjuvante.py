from math import ceil, pow, sqrt
import numpy as np
import openml
import pandas as pd
import time
from openml.datasets import edit_dataset, fork_dataset, get_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

#Estabelecendo porcentagem de treino
#-----------------------------------------------------------------------

trainPercentage = 0.50

if trainPercentage > 1:
    print("PORCENTAGEM SÓ VAI ATÉ UM, SEU BURRO!!!!!!!!")
    exit

# Adquirindo e processando o dataset de LoL
# Aqui, tentamos predizer as vitórias e derrotas baseado nas características de cada time
#-----------------------------------------------------------------------
dataset = openml.datasets.get_dataset(43635)

X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="array", target=dataset.default_target_attribute)

#estabelecendo que as classes são a vitória
#0=derrota do azul
#1=vitória do azul
df = pd.DataFrame(X, columns=attribute_names)
df['class'] = df['blue_win']

#dropando colunas inúteis
df = df.drop(["Unnamed:_0", "matchId", "blue_win"], axis=1)

#não existem instâncias com valores nulos nesse dataset
#Logo, não é necessário imputar dados

p= MaxAbsScaler()
p.fit(df)

X = df.values
y = df['class'].values

X = np.delete(X, 16, axis=1)

#Distribuição igualitária dos resultados
#-----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-trainPercentage, random_state=0)

#Knn Classificador
#-----------------------------------------------------------------------

k = 3

if k%2 == 0:
    print("WARNING: Para evitar empates, um valor ímpar é sempre recomendado!")

yteste = []
iterat = 0

print("Iniciando processo")
# ti = Tempo Inicial
ti = time.time()

#Para cada valor de teste, existe uma lista de distâncias
for teste in X_test:
    cont = 0
    listaDistancia = []
    #calcula as distâncias euclidianas de todas as amostras de treino
    for treino in X_train:
        dist = np.linalg.norm(teste-treino)
        if len(listaDistancia) < k:
            listaDistancia.append((dist,cont))
        elif max(listaDistancia, key=lambda tup: tup[0])[0] > dist:
            listaDistancia.remove(max(listaDistancia, key=lambda tup: tup[0]))
            listaDistancia.append((dist, cont))
        cont += 1
    #pega os k menores valores
    aux1 = 0 #0
    aux2 = 0 #1
    #(distancia, indice)
    for tupla in listaDistancia:
        if y_train[tupla[1]] == 0:
            aux1 = aux1 + 1
        elif y_train[tupla[1]] == 1:
            aux2 = aux2 + 1
    if aux1 > aux2:
        yteste.append(0)
    elif aux2 > aux1:
        yteste.append(1)
    else:
        print("Seu K está errado, caramba!")
    iterat = iterat + 1

# tf = Tempo Final
tf = time.time()
print("Fim do processo")


#Cálculo das Métricas de acurácia e matriz de confusão
#-----------------------------------------------------------------------

TP = 0
TN = 0
FP = 0
FN = 0
controle = 0


# 0 == positivo
# 1 == negativo
for result in y_test:
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
matrizConf = [[TP, FP], [FN, TN]]

print("Tempo de Processamento(s): " + str(tf-ti))
print("Matriz de Confusão")
print(matrizConf)
print("Amostras de treino: " + str(trainPercentage*100) + "%")
print("Valor de K: " + str(k))
print(str(ACC*100) + "% de acurácia")