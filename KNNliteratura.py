from math import ceil
import numpy
import openml
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import confusion_matrix, accuracy_score
from openml.datasets import get_dataset

# Adquirindo e processando o dataset
#-----------------------------------------------------------------------
dataset = openml.datasets.get_dataset(50)

X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="array", target=dataset.default_target_attribute)

#Estabelecendo padrão da porcentagem de treino
#-----------------------------------------------------------------------

trainPercentage = 0.80

if trainPercentage > 1:
    print("PORCENTAGEM SÓ VAI ATÉ UM, SEU BURRO!!!!!!!!")
    exit

indice = ceil(X.shape[0]*trainPercentage)

#Distribuição igualitária dos resultados
#-----------------------------------------------------------------------
unique, counts = numpy.unique(y, return_counts=True)

indice0 = ceil(counts[1]*trainPercentage) #% de uns
indice1 = ceil(counts[0]*trainPercentage) #% de zeros

Xtreino = X[:indice0].tolist() + X[counts[1]:indice1].tolist()
Ytreino = y[:indice0].tolist() + y[counts[1]:indice1].tolist()
Xteste = X[indice0:counts[1]].tolist() + X[indice1:].tolist()
Yvalida = y[indice0:counts[1]].tolist() + y[indice1:].tolist()

#Knn Classificador
#-----------------------------------------------------------------------

knnClass = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
knnClass.fit(Xtreino, Ytreino)

yteste = knnClass.predict(Xteste)

#Cálculo das Métricas de acurácia e matriz de confusão
#-----------------------------------------------------------------------

matrizConf = confusion_matrix(Yvalida, yteste)
ACC = accuracy_score(Yvalida, yteste)

#Print resultados finais
#-----------------------------------------------------------------------

print("Matriz de Confusão")
print(matrizConf)
print("Usando " + str(trainPercentage*100) + "% das amostras para treino ")
print(str(ACC*100) + "% de acurácia")