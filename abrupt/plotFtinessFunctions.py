import numpy as np
import os
import pandas as pd
import time
from multiprocessing import Pool
import itertools
import math
import matplotlib.pyplot as plt
import pickle
import ternary
import seaborn as sns
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sympy.utilities.iterables import multiset_permutations
from decimal import Decimal as D
import operator
from scipy.stats import binom
import scipy.special


beta = 0.2  # beta determines the curvature of the fitness and reward functions
T = 20
tArr = np.arange(0,T+1,1)
linear = lambda x : x
def increasing(beta,x,T):
    alphaRI = T / ((np.exp(beta * T)) - 1)
    phiVar = alphaRI * (np.exp(beta * x) - 1)
    return phiVar

def diminishing(beta,x,T):
    alphaRD = T / (1 - (np.exp(-beta*T)))
    print(alphaRD)
    phiVar = alphaRD * (1 - np.exp(-beta*x))
    return phiVar

# Data for plotting
linearData = linear(tArr)
increasingData = [increasing(beta,t,T) for t in tArr]

diminishingData = [diminishing(beta,t,T) for t in tArr]

fig, ax = plt.subplots()
ax.plot(tArr, linearData, label = 'linear')
ax.plot(tArr, diminishingData, label = 'diminishing')

ax.plot(tArr, increasingData, label = 'increasing')

ax.set(xlabel='Number of specialization', ylabel='Fitness contribution',
       title='Fitness mappings')
ax.grid()
plt.legend(loc="upper left")
fig.savefig("fitnessFunctions.png")
plt.show()