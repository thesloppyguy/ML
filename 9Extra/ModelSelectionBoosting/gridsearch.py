from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("DATA/Social_Network_Ads.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

svm = SVC(kernel='rbf', random_state=0)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracies = cross_val_score(estimator=svm, X=X_train, y=y_train, cv=10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

parameters = [{"C": [0.25, 0.5, 0.75, 1], "kernel": ['linear']},
              {"C": [0.25, 0.5, 0.75, 1], "kernel": ['rbf'], "gamma": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]


gsearch = GridSearchCV(
    estimator=svm, param_grid=parameters, scoring='accuracy', cv=10, n_jobs=-1)

gsearch.fit(X_train, y_train)
best_score = gsearch.best_score_
best_combo = gsearch.best_params_
print(best_score)
print(best_combo)
