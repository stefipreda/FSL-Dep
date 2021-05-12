import dynet as dy
import numpy as np
import math


def get_loss(u, sign):
    if sign == 1:
        # From the data
        return -dy.log_sigmoid(u)
    else:
        return -dy.log_sigmoid(-u)


def probability_distribution(count_table, alpha):
    count_all = 0
    for tok in count_table.keys():
        count_all += count_table[tok] ** alpha
    probs = []
    for tok in count_table.keys():
        probs.append((count_table[tok] ** alpha) / count_all)
    return np.array(probs)


# Slow sampling . For fast sampling, use alias method below
def numpy_sample(count, probs):
    return np.random.choice(len(probs), count, p=probs)

""" Alias method from https://github.com/guyemerson/sem-func/blob/master/src/core/utils.py"""

def init_alias(prob):
    """
    Initialise arrays for sampling with the alias method
    :param prob: probability array
    :return: probability table, alias table
    """
    N = prob.size
    # Initialise tables
    U = prob.astype('float64') / prob.sum() * N
    K = np.arange(N)
    # Initialise lists with weight above and below 1
    below = [i for i, x in enumerate(U) if x < 1]
    above = [i for i, x in enumerate(U) if x >= 1]
    # Fill tables
    # In each iteration, we remove one index from the pair of lists
    while above and below:
        # Take a pair of indices, one above and one below
        i = below.pop()
        j = above.pop()
        # Fill in the tables
        K[i] = j
        # Calculate the remaining weight of j, and put it back in the correct list
        U[j] -= (1 - U[i])
        if U[j] < 1:
            below.append(j)
        else:
            above.append(j)
    # Note the final index will have U=1, up to rounding error
    return U, K


def alias_sample_one(U, K):
    """
    Sample from a categorical distribution, using the alias method
    :param U: probability table
    :param K: alias table
    :return: sample
    """
    # Choose a random index
    i = np.random.randint(U.size)
    # Return the index, or the alias
    if np.random.rand() > U[i]:
        return K[i]
    else:
        return i


def alias_sample(U, K, n=None):
    """
    Sample from a categorical distribution, using the alias method
    :param U: probability table
    :param K: alias table
    :param n: number of samples to draw (int or tuple of ints)
    :return: array of samples
    """
    if n:
        # Choose random indices
        i = np.random.randint(U.size, size=n)
        # Choose whether to return indices or aliases
        switch = (np.random.random(n) > U[i])
        return switch * K[i] + np.invert(switch) * i
    else:
        return alias_sample_one(U, K)
