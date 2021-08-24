import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('DATA/Ads_CTR_Optimisation.csv')
N = 10000
a = []
b = []
for i in range(0, 10):
    for j in range(1, N):
