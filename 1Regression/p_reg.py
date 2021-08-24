from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("DATA/Position_Salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# no need to split the the set if you have less data just use all of it

# apply linear re
lr = LinearRegression()
lr.fit(x, y)

pf = PolynomialFeatures(degree=4)
x_poly = pf.fit_transform(x)

lr_2 = LinearRegression()
lr_2.fit(x_poly, y)

# graph for linear regression
plt.scatter(x, y, color='red')
plt.plot(x, lr.predict(x), color='blue')
plt.title("LinearRegression")
plt.xlabel("level")
plt.ylabel("Salary")
plt.show()

# graph for polynomial regression
plt.scatter(x, y, color="red")
plt.plot(x, lr_2.predict(x_poly), color="blue")
plot.title("PolynomialLinearRegression_v2")
plot.xlabel("level")
plot.ylabel("salary")
plt.show()

# smooth graph for polynomial regression not tought as you will get a lot of data
X_grid = np.arange(min(x), max(x), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color='red')
plt.plot(X_grid, lr_2.predict(pf.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


print(lr.predict([[6.5]]))

# convert it to polynomial from then apply predict
print(lr_2.predict(pf.fit_transform([[6.5]])))
