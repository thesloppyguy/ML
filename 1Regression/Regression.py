from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("DATA\BigDATA.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# #dealing with empty data
# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# imputer.fit(x[:, 1:3])
# x[:, 1:3] = imputer.transform(x[:, 1:3])

# #making dummy variable
# ct = ColumnTransformer(
#     transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# x = np.array(ct.fit_transform(x))

# spliting the data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

y_train = y_train.reshape(len(y_train), 1)
y_test = y_test.reshape(len(y_test), 1)

# feature scaling
sc_x = StandardScaler()
sc_y = StandardScaler()
x_train_fs = sc_x.fit_transform(x_train)
y_train_fs = sc_y.fit_transform(y_train)


# multiple linear regression
mlr = LinearRegression()
mlr.fit(x_train, y_train)
y_pred_multi = mlr.predict(x_test)


# polynomial linear regression
pf = PolynomialFeatures(degree=4)
x_poly = pf.fit_transform(x_train)
x_poly_test = pf.transform(x_test)
plr = LinearRegression()
plr.fit(x_poly, y_train)
y_pred_poly = plr.predict(x_poly_test)


# support vector linear regression
regressor = SVR(kernel='rbf')
regressor.fit(x_train_fs, y_train_fs)
y_pred_sr = sc_y.inverse_transform(regressor.predict(sc_x.transform(x_test)))

# decision tree regression
dt = DecisionTreeRegressor(random_state=0)
dt.fit(x_train, y_train)
y_pred_dt = dt.predict(x_test)


# random forest regression
rf = RandomForestRegressor(n_estimators=10, random_state=0)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)


# r2 score
print("multi :"+str(r2_score(y_test, y_pred_multi)))
print("poly :"+str(r2_score(y_test, y_pred_poly)))
print("support vector :"+str(r2_score(y_test, y_pred_sr)))
print("decision tree :"+str(r2_score(y_test, y_pred_dt)))
print("random forest :"+str(r2_score(y_test, y_pred_rf)))
