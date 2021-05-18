from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
import pandas as pd

# data pre prossing
dataset = pd.read_csv('DATA/Churn_Modelling.csv')
x = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])

ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

# we always apply feature scaling in deep learing and to all features regardless dummy or not
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# building a ANN
ann = tf.keras.models.Sequential()
# initializing first hidden layer
# relu = Rectifier function
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# adding another layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# adding output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# training the ANN
# compleing ANN
# adam = stotastic gradient decent # binary_crossentropy = 0/1 # categorical creessentropy = more than binary
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# batch size for batch processing # epoch is for the number of cycles of stocastic gradient decent to apply on the
ann.fit(x_train, y_train, batch_size=32, epochs=100)

K = ann.predict(sc.transform(
    [[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))

if K > 0.5:
    print("pass")
else:
    print("remove")

y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test, y_pred)
# print(cm)

ac = accuracy_score(y_test, y_pred)
print(ac)
