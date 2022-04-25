# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 20:22:01 2022

@author: Robert
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import time
from tangent_parameterization import getPoints

# essentially all of this will belong to the Domain class;

# move to the domain class
def _rho(pairs, theta_k):
    pt = 0
    n = 0
    for pair in pairs:
        a,b = pair.split(',')
        a = float(a)
        b= float(b)
        pt += ((a * np.sin(n*theta_k)) + (b * np.cos(n*theta_k)))
        n += 1
    return pt

# use more expressive names
def _calc_part_1(x_list, y_list, theta_list, k):
    q = len(x_list)

    return ((((x_list[k] - x_list[(k-1)%q]) * np.cos(theta_list[k])) +
             ((y_list[k] - y_list[(k-1)%q]) * np.sin(theta_list[k]))) /
            (math.sqrt((x_list[k] - x_list[(k-1)%q])**2 +
                       (y_list[k] - y_list[(k-1)%q])**2)))

# use more expressive names
def _calc_part_2(x_list, y_list, theta_list, k):
    q = len(x_list)

    return ((((x_list[(k+1)%q] - x_list[k]) * np.cos(theta_list[k])) +
         ((y_list[(k+1)%q] - y_list[k]) * np.sin(theta_list[k]))) /
            (math.sqrt((x_list[(k+1)%q] - x_list[k])**2 +
                       (y_list[(k+1)%q] - y_list[k])**2)))

def calc_dLq_over_dTheta_k(pairs, x_list, y_list, theta_list, k):
    return _rho(pairs, theta_list[k]) * (_calc_part_1(x_list, y_list, theta_list, k) - _calc_part_2(x_list, y_list, theta_list, k))


def _check_neg(point1, point2):
    if point1 - point2 < 0:
        return point1 + (2*math.pi)
    else:
        return point1

'''
Constructs gradient in the form of:

    np.array([np.cos((_check_neg(t[0], t[3]) - t[3])/2) - np.cos((_check_neg(t[1], t[0]) - t[0])/2),
              np.cos((_check_neg(t[1], t[0]) - t[0])/2) - np.cos((_check_neg(t[2], t[1]) - t[1])/2),
              np.cos((_check_neg(t[2], t[1]) - t[1])/2) - np.cos((_check_neg(t[3], t[2]) - t[2])/2),
              np.cos((_check_neg(t[3], t[2]) - t[2])/2) - np.cos((_check_neg(t[0], t[3]) - t[3])/2),
              ...
              ])

    where the first column of t's is "first", the second column is "second, ...

'''

# Gradient of what?
#
# This is the gradient of the q-length function.
def _gradient(pairs, t_theta_list):
    q = len(t_theta_list)
    x_list2, y_list2 = getPoints(pairs, t_theta_list)

    return np.array([calc_dLq_over_dTheta_k(pairs, x_list2, y_list2, t_theta_list, k) for k in range(q)])

def gradient_ascent(pairs, initial_condition, learn_rate, n_iter = 10000, tolerance=1e-08):
    vector = initial_condition

    for i in range(n_iter):
        diff = _gradient(pairs, vector)
        if np.all(np.abs(diff) <= tolerance):
            break
        vector += diff * learn_rate
    return vector
