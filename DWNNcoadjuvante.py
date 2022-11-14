from math import ceil
import numpy as np
import pandas as pd
import openml
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from openml.datasets import get_dataset
import time
  

#Estabelecendo porcentagem de treino
#-----------------------------------------------------------------------

trainPercentage = 0.90

if trainPercentage > 1:
    print("PORCENTAGEM SÓ VAI ATÉ UM, SEU BURRO!!!!!!!!")
    exit

# Adquirindo e processando o dataset de Operações de Uma CPU
# Aqui, tentamos predizer as vitórias e derrotas baseado nas características de cada time
#-----------------------------------------------------------------------
dataset = openml.datasets.get_dataset(562)

X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="array", target=dataset.default_target_attribute)

#estabelecendo que as classes são a % de tempo que o CPU roda em modo de usuário
df = pd.DataFrame(X, columns=attribute_names)
df['class'] = y

#não existem instâncias com valores nulos nesse dataset
#Logo, não é necessário imputar dados

p= MaxAbsScaler()
p.fit(df)

X = df.values
y = df['class'].values

#Distribuição igualitária dos resultados
#-----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-trainPercentage, random_state=0)

#Estabelecendo K
#-----------------------------------------------------------------------
k = 5

if k%2 == 0:
    print("WARNING: Para evitar empates, um valor ímpar é sempre recomendado!")

#Knn Regressor (DWNN)
#-----------------------------------------------------------------------

k = 5

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
            listaDistancia.append((dist,cont,1/(dist**2)))
        elif max(listaDistancia, key=lambda tup: tup[0])[0] > dist:
            listaDistancia.remove(max(listaDistancia, key=lambda tup: tup[0]))
            listaDistancia.append((dist, cont, 1/(dist**2)))
        cont += 1
    #(distancia, indice, peso)
    #Soma todos os pesos
    somaNum = 0
    somaDen = 0
    for tupla in listaDistancia:
        somaNum += tupla[2]*y_train[tupla[1]]
        somaDen += tupla[2]
    yteste.append(somaNum/somaDen)
    iterat = iterat + 1

# tf = Tempo Final
tf = time.time()
print("Fim do processo")

#Cálculo da Métrica de erro absoluto médio
#-----------------------------------------------------------------------

MAE = mean_absolute_error(y_test, yteste)

#Print resultados finais
#-----------------------------------------------------------------------

print("Tempo de Processamento(s): " + str(tf-ti))
print("Amostras de treino: " + str(trainPercentage*100) + "%")
print("Valor de K: " + str(k))
print(str(MAE) + "% de erro absoluto médio")