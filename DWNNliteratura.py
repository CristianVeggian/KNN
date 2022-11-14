from math import ceil
import numpy as np
import pandas as pd
import openml
import time
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from openml.datasets import get_dataset

arq = open('resultadosDWNNliteratura.txt', 'a')

# Adquirindo e processando o dataset de Operações de uma CPU
# Aqui, tentamos predizer o tempo de CPU consumido pelo usuário baseado
# nas chamadas e na ordem de operações
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

X_total = df.values
y_total = df['class'].values

for k in range(3,13,2):
    for pctg in range(50,100,10):

        trainPercentage = pctg/100

        #Distribuição igualitária dos resultados
        #-----------------------------------------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-trainPercentage, random_state=0)

        #Knn Regressor (DWNN)
        #-----------------------------------------------------------------------

        print("Iniciando processo")
        # ti = Tempo Inicial
        ti = time.time()

        knnClass = KNeighborsRegressor(n_neighbors=k, metric="euclidean", weights='distance')
        knnClass.fit(X_train, y_train)

        yteste = knnClass.predict(X_test)

        # tf = Tempo Final
        tf = time.time()
        print("Fim do processo")

        #Cálculo da Métrica de erro absoluto médio
        #-----------------------------------------------------------------------

        MAE = mean_absolute_error(y_test, yteste)

        #Print resultados finais
        #-----------------------------------------------------------------------

        arq.write("\n" + str(k))
        arq.write(" - " + str(trainPercentage*100))
        arq.write(" - " + str(MAE).replace('.',','))
        arq.write(" - " + str(tf-ti).replace('.',','))

        print("Tempo de Processamento(s): " + str(tf-ti))
        print("Amostras de treino: " + str(trainPercentage*100) + "%")
        print("Valor de K: " + str(k))
        print(str(MAE) + "% de erro absoluto médio")

arq.close()