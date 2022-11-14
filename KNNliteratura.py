from math import ceil
import numpy as np
import pandas as pd
import openml
import time
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from openml.datasets import get_dataset

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

p = MaxAbsScaler()
p.fit(df)

#não existem instâncias com valores nulos nesse dataset
#Logo, não é necessário imputar dados

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

        # ti = Tempo Inicial
        ti = time.time()

        knnClass = KNeighborsClassifier(n_neighbors=k, metric="euclidean", algorithm='ball_tree')
        knnClass.fit(X_train, y_train)

        yteste = knnClass.predict(X_test)

        # tf = Tempo Final
        tf = time.time()

        #Cálculo das Métricas de acurácia e matriz de confusão
        #-----------------------------------------------------------------------

        matrizConf = confusion_matrix(y_test, yteste)
        ACC = accuracy_score(y_test, yteste)

        #Print resultados finais
        #-----------------------------------------------------------------------

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