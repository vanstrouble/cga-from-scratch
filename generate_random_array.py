# Test to generate random array with weights. NumPy vs Random

import numpy as np
from random import choices


def fill_np_random(p):
    a = (np.random.random(size=len(p)) < p).astype(int)
    return a


def fill_np_binomial(p):
    a = np.random.binomial(1, p, len(p))
    return a


def fill_random_array(p):
    a = np.asarray([choices([0, 1], weights=[1 - prob, prob])[0] for prob in p])
    return a


p = np.random.rand(4)
# p = np.array([1, 1, 1, 1])
# p = np.array([0.1, 0.3, 0.6, 0.9])
# p = np.array([1.24, 0.3, 0.6, 0.9])

print('NumPy Array')
for i in range(10):
    np.set_printoptions(2)
    print(f'{i + 1} - \tp: {p}, a: {fill_np_random(p)}')

print('\nNumPy Binomial Array')
for i in range(10):
    np.set_printoptions(2)
    print(f'{i + 1} - \tp: {p}, a: {fill_np_binomial(p)}')

print('\nRandom Array')
for i in range(10):
    np.set_printoptions(2)
    print(f'{i + 1} - \tp: {p}, a: {fill_random_array(p)}')
