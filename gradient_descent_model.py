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

def _calc_part_1(x_list, y_list, theta_list, k):
    if k == 0:
        k_minus_1 = len(x_list) - 1
    else:
        k_minus_1 = k - 1
        
    if k == len(x_list) - 1:
        k_plus_1 = 0
    else:
        k_plus_1 = k + 1
        
    return (
        (((x_list[k] - x_list[k_minus_1]) * np.cos(theta_list[k])) + ((y_list[k] - y_list[k_minus_1]) * np.sin(theta_list[k]))) 
            / 
            (math.sqrt((x_list[k] - x_list[k_minus_1])**2 + (y_list[k] - y_list[k_minus_1])**2))
            )

def _calc_part_2(x_list, y_list, theta_list, k):
    if k == len(x_list) - 1:
        k_plus_1 = 0
    else:
        k_plus_1 = k + 1
        
    return (
        (((x_list[k_plus_1] - x_list[k]) * np.cos(theta_list[k])) + ((y_list[k_plus_1] - y_list[k]) * np.sin(theta_list[k]))) 
            / 
            (math.sqrt((x_list[k_plus_1] - x_list[k])**2 + (y_list[k_plus_1] - y_list[k])**2))
            )

def calc_dLq_over_dTheta_k(x_list, y_list, theta_list, k, pairs):
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
def _gradient(t_theta_list, pairs):
    q = len(t_theta_list)
    x_list2, y_list2 = getPoints(pairs, t_theta_list)
    
    lq = []
    for k in range(q):
        lq += [calc_dLq_over_dTheta_k(x_list2, y_list2, t_theta_list, k, pairs)]
        
    return np.array(lq)

def gradient_ascent(start, pairs, learn_rate, n_iter = 10000, tolerance=1e-08):
    vector = start
    
    for i in range(n_iter):
        #print(_gradient(vector))
        diff = learn_rate * _gradient(vector, pairs)
        if np.all(np.abs(diff) <= tolerance):
            break
        vector += diff
    return vector


if __name__ == "__main__":
    points = gradient_ascent(
         start=np.array([0.0,math.pi/2.0,math.pi,3*math.pi/2.0]), learn_rate=0.001
    )
    
    l = points.tolist()
    for item in l:
        print(item/(2*math.pi) % 1)
  
    
    
    

'''
RETIRED GRADIENT DESCENT WITH HARD CODED GRADIENT ARRAY

def gradient_descent(gradient, start, learn_rate, n_iter = 5, tolerance=1e-08):
    vector = start
    
    for _ in range(n_iter):
        print(gradient(vector))
        diff = -learn_rate * gradient(vector)
        print(diff)
        if np.all(np.abs(diff) <= tolerance):
            break
        vector += diff
    return vector

points = gradient_descent(
     gradient=lambda t: np.array([np.cos((_check_neg(t[0], t[3]) - t[3])/2) - np.cos((_check_neg(t[1], t[0]) - t[0])/2), 
                                  np.cos((_check_neg(t[1], t[0]) - t[0])/2) - np.cos((_check_neg(t[2], t[1]) - t[1])/2), 
                                  np.cos((_check_neg(t[2], t[1]) - t[1])/2) - np.cos((_check_neg(t[3], t[2]) - t[2])/2), 
                                  np.cos((_check_neg(t[3], t[2]) - t[2])/2) - np.cos((_check_neg(t[0], t[3]) - t[3])/2)
                                  ]),
     start=np.array([0.0,math.pi/2.0,math.pi,3*math.pi/2.0]), learn_rate=0.001
)

l = points.tolist()
for item in l:
    print(item/(2*math.pi) % 1)
    
'''

    
