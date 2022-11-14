from math import ceil, pow, sqrt
import numpy as np
import openml
import pandas as pd
import time
from openml.datasets import edit_dataset, fork_dataset, get_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.neighbors import BallTree

arq = open("resultados.txt", "a")

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

X_total = df.values
y_total = df['class'].values

X_total = np.delete(X_total, 16, axis=1)

p= MaxAbsScaler()
p.fit(df)

#Não existem instâncias com valores nulos nesse dataset
#logo, não é necessário imputar dados

#Estabelecendo porcentagens de treino e k's
#-----------------------------------------------------------------------
for k in range(3,13,2):
    for pctg in range(50,100,10):
        trainPercentage = pctg/100

        #Distribuição igualitária dos resultados
        #-----------------------------------------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(X_total, y_total, test_size=1-trainPercentage, random_state=0)

        #Knn Classificador
        #-----------------------------------------------------------------------

        yteste = []

        arvore = BallTree(X_train, leaf_size=10, metric='euclidean')
        dist, ind = arvore.query(X_test, k=k)  

        # ti = Tempo Inicial
        ti = time.time()

        #Para cada valor de teste, existe uma lista de distâncias
        for indice in ind:
            #pega os k menores valores
            aux1 = 0 #0
            aux2 = 0 #1
            #(distancia, indice)
            for valor in indice:
                if y_train[valor] == 0:
                    aux1 = aux1 + 1
                elif y_train[valor] == 1:
                    aux2 = aux2 + 1
            if aux1 > aux2:
                yteste.append(0)
            elif aux2 > aux1:
                yteste.append(1)
            else:
                print("Seu K está errado, caramba!")

        # tf = Tempo Final
        tf = time.time()

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
        matrizConf = [[TP, FN], [FP, TN]]

        arq.write("\n" + str(k))
        arq.write(" - " + str(trainPercentage*100))
        arq.write(" - " + str(ACC).replace('.',','))
        arq.write(" - " + str(tf-ti).replace('.',','))
        
        print("Tempo de Processamento(s): " + str(tf-ti).replace('.',','))
        print("Matriz de Confusão")
        print(matrizConf)
        print("Amostras de treino: " + str(trainPercentage*100) + "%")
        print("Valor de K: " + str(k))
        print(str(ACC).replace('.',',') + "% de acurácia")

arq.close()