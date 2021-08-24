from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("DATA/Position_Salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

rf = RandomForestRegressor(n_estimators=10, random_state=0)
rf.fit(x, y)


x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color="red")
plt.plot(x_grid, rf.predict(x_grid), color="blue")
plt.title("Decision tree")
plt.xlabel("level")
plt.ylabel("salary")
plt.show()
