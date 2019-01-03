# Cross-entropy Quiz
# This Script does not Work
# Default format created by Udacity

import numpy as np


# Quiz : Write a function retursn the float corresponding to their cross-entropy.

# ==Answer==
def cross_entropy(Y, P):
    Y = np.asarray(Y).astype(float)
    P = np.asarray(P).astype(float)

    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))


# ==Answer==


"""
Many way to change list to float

1. Change list to array and change type float[np.asarray -> X.astype(float)]
2. Change list to array and set type float [np.asarray(a, dtype=np.float64)]
3. Using np.float_
"""