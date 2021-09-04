from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import csv

train = pd.read_csv("practice/train_data.csv")
x_train = train.iloc[:, :-1].values
y_train = train.iloc[:, -1].values

test = pd.read_csv("practice/test_data.csv")
x_test = test.iloc[:, :].values

y_train = y_train.reshape(len(y_train), 1)

sc = StandardScaler()
x_train[:, :] = sc.fit_transform(x_train[:, :])
x_test[:, :] = sc.transform(x_test[:, :])

lr = LinearRegression()
lr.fit(x_train, y_train)

y_test = lr.predict(x_test)
np.set_printoptions(precision=2)

with open('practice/result.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(list[y_test])
