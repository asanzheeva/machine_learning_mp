import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# евклидова метрика - векторы
# алгоритм KNN
# нормировать данные - схлопнуть в отрезок (0;1) или (-1;1) - (Xi - Xmin)/mакс эл.


iris_csv = pd.read_csv("iris.csv")
print(iris_csv)

# стандартизация датасета
scaler = StandardScaler() 
scaler.fit(iris_csv.drop('variety', axis = 1)) # обучение датасета 
scaled_features = scaler.transform(iris_csv.drop('variety', axis = 1)) # стандартизация признаков
scaled_data = pd.DataFrame(scaled_features, columns = iris_csv.drop('variety', axis = 1).columns) 

# разделение набора данных на:
# train - обучающая выборка, test - тестовая выборка
x = scaled_data
y = iris_csv['variety']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, stratify = y,random_state = 11) # X и y – это исходные данные и метки классов,
# stratify - разделение на группы, теперь в каждой из выборок будет одинаковое соотношение классов


# обучение модели K-ближайших соседей
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(x_train, y_train)
predictions = knn.predict(x_test)

#print(classification_report(y_test, predictions))

print(accuracy_score(y_test, predictions))

print(confusion_matrix(y_test, predictions))
