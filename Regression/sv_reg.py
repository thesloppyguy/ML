from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib as plt
import pandas as pd

dataset = pd.read_csv("DATA/Position_Salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(len(y), 1)


x_sc = StandardScaler()
y_sc = StandardScaler()
x = x_sc.fit_transform(x)
y = y_sc.fit_transform(y)


sv = SVR(x)
# save for later
