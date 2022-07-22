import numpy as np
from regex import P
from bandit_base import *
import scipy.optimize as opt
from D_PBL import PBL

from bandit_base import GLUCB


PBL = GLUCB()

# # Test epsilon greedy strategy
# empiricalMeans, V, n_a = ucb.epsilonGreedyBandit(nIterations=200)
# print("\nepsilonGreedyBandit results")
# print(empiricalMeans)
# print(V)
# print(n_a)


# # Test UCB strategy
# empiricalMeans,V , n_a= ucb.UCBbandit(nIterations=200)
# print("\nUCBbandit results")
# print(empiricalMeans)
# print(V)
# print(n_a)

if __name__ == "__main__":
    print(PBL.D_GLUCB(iter=400))