# Softmax_Fucntion Quiz
# This Script does not Work
# Default format created by Udacity

import numpy as np

# Quiz : Write a function that takes as input a list of numbers, and returns the list of values given by the softmax function.

# ==Answer==
def softmax(L):
    expL = np.exp(L)
    sum_expL = sum(expL)
    result_list = []
    for i in expL:
        result_list.append(i / sum_expL)

    return result_list
# ==Answer==