from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np  # for arrays
import matplotlib.pyplot as plt  # for graphs
import pandas as pd  # for matrixs
# opencv-python==4.1.0.25

# imporing data sets
data_set = pd.read_csv('DATA\Data.csv')
# in machine learning you get a dependent variable (what to find) 'Y' and feature (from which we need to find the dependent cariable) 'X'
x = data_set.iloc[:, :-1].values  # row, column  # : = range # -1 = last value
# this format is only valid if the given data is organised ie last column is 'Y' and rest 'X'
y = data_set.iloc[:, -1].values

# print(x)
# print(y)


imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
# print(x)

ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
# print(x)


le = LabelEncoder()
y = le.fit_transform(y)
# print(y)

# splitting the data into test and train sets
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)


# feature scaling
# sometimes used sometimes not
# why ????
# normalzation for normal sets
# standardization works all the time
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print(X_train)
print(X_test)
