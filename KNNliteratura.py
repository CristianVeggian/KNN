from math import ceil
import numpy as np
import pandas as pd
import openml
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from openml.datasets import get_dataset

#Estabelecendo porcentagem de treino
#-----------------------------------------------------------------------

trainPercentage = 0.6

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

knnClass = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
knnClass.fit(X_train, y_train)

yteste = knnClass.predict(X_test)

#Cálculo das Métricas de acurácia e matriz de confusão
#-----------------------------------------------------------------------

matrizConf = confusion_matrix(y_test, yteste)
ACC = accuracy_score(y_test, yteste)

#Print resultados finais
#-----------------------------------------------------------------------

print("Matriz de Confusão")
print(matrizConf)
print("Usando " + str(trainPercentage*100) + "% das amostras para treino ")
print(str(ACC*100) + "% de acurácia")