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

def calc_dLq_over_dTheta_k(d, p, θ, k):
    q = len(p)

    Δx0 = (p[k][0] - p[(k-1)%q][0])
    Δy0 = (p[k][1] - p[(k-1)%q][1])
    ℓ0 = math.sqrt(Δx0**2+Δy0**2)

    Δx1 = (p[(k+1)%q][0] - p[k][0])
    Δy1 = (p[(k+1)%q][1] - p[k][1])
    ℓ1 = math.sqrt(Δx1**2+Δy1**2)

    cosθ = np.cos(θ[k])
    sinθ = np.sin(θ[k])

    return d.ρ(θ[k]) * ((Δx0*cosθ+Δy0*sinθ)/ℓ0 - (Δx1*cosθ+Δy1*sinθ)/ℓ1)

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
def _gradient(d, θ):
    q = len(θ)
    p_list = list(map(d.γ, θ))

    return np.array([calc_dLq_over_dTheta_k(d, p_list, θ, k) for k in range(q)])

def gradient_ascent(d, initial_condition, learn_rate, n_iter = 10000, tolerance=1e-08):
    vector = initial_condition

    for i in range(n_iter):
        diff = _gradient(d, vector)
        if np.all(np.abs(diff) <= tolerance):
            break
        vector += diff * learn_rate
    return vector
