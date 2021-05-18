from apyori import apriori
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('DATA/Market_Basket_Optimisation.csv', header=None)
# we make a list of transactions from the csv file to a python object which we can modify
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

# use common sense for support, try different values for confidence, lift min is 3 and everything above is good, lengths set the relation members
rules = apriori(transactions=transactions, min_support=0.003,
                min_confidence=0.2, min_lift=3, min_length=2, max_length=2)

results = list(rules)


def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))


resultsinDataFrame = pd.DataFrame(inspect(results), columns=[
                                  'Product 1', 'Product 2', 'Support'])

print(resultsinDataFrame)

print(resultsinDataFrame.nlargest(n=10, columns='Support'))
