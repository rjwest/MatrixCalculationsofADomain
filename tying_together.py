# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 19:14:43 2022

@author: Robert
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import time

from gradient_descent_model import gradient_ascent
from tangent_parameterization import getPoints

from domain import Domain

# this computes the first variation of the q-length with respect to e_k
def _fetch_val_matrix_Anq(d,Θ, k, q):
    summation = 0
    for j in range(q):
        # also this can be optimized: each point is computed twice.
        # change names for x_list and y_list below
        x_list,y_list = getPoints(d.pairs, [Θ[j], Θ[(j + 1) % q]])

        α_j = np.arctan2(y_list[1] - y_list[0], x_list[1] - x_list[0])

        summation += d.e_k(Θ[j], k) * np.sin(( α_j - Θ[j]))
    return summation

# this computes all elements of the matrix.
def gen_matrix_Anq(d, N):
    theta_list = []
    matrix = np.array([[]])
    non = 0

    # each row is independent on the other rows; each could be computed
    # in a separate thread
    for q in range(2, (N+2), 1):
        # Set up uniform (equispaced) initial conditions for gradient
        # ascent Note that since the initial condition is symmetric
        # with respect to 0, so will be the solution.
        start = time.time()

        # guess initial conditions to be equispaced in Lazutkin coords
        θ_guess=[d.inverse_Lazutkin(x) for x in np.arange(0,1,1/q)]

        # find the orbit of rotation number 1/q
        Θ = gradient_ascent(d, θ_guess)

        #Generate each index of a row in the matrix
        partial = time.time()
        row = np.array([_fetch_val_matrix_Anq(d, Θ, k, q) for k in range(2, N+2)])
        end = time.time()
        print(f'{q}-periodic orbit found in {partial-start}s, Row {q-1} computed in {end-partial}s')
        if non == 0:
            matrix = [row]
            non = 1
        else:
            matrix = np.append(matrix, [row], axis=0)

    return matrix

if __name__ == "__main__":



    #BUILD THE ORIGINAL DOMAIN
    d = Domain('coeff1.txt')

    print(d.pairs)

    orig_q = 201
    orig_ep = 2*math.pi/orig_q
    theta_list_domain = [theta for theta in np.arange(0,(orig_q*orig_ep) + orig_ep,orig_ep)]

    x_list_domain, y_list_domain = getPoints(d.pairs, theta_list_domain)

    #plt.axes().set_aspect('equal')
    #plt.plot(x_list_domain,y_list_domain)

    N = 200
    matrix = gen_matrix_Anq(d, N)

    temp = np.linalg.eigvals(matrix)
    x_list = [ele.real for ele in temp]
    y_list = [ele.imag for ele in temp]

    max_x = max(x_list)
    min_x = min(x_list)

    max_y = max(y_list)
    min_y = min(y_list)

    print('MATRIX COMPUTED')
    #print(matrix)

    for n in range(2,N):
        #print(matrix[0:n,0:n])
        inner_matrix = matrix[0:n,0:n]
        eigenvals = np.linalg.eigvals(inner_matrix)

        # extract real part
        x = [ele.real for ele in eigenvals]
        # extract imaginary part
        y = [ele.imag for ele in eigenvals]

        plt.figure()
        plt.xlim(min_x - 0.5, max_x + 0.5)
        plt.ylim(min_y - 0.5, max_y + 0.5)
        plt.scatter(x, y)

        plt.ylabel('Imaginary')
        plt.xlabel('Real')
        plt.savefig(f'eigens//eigens_{n}.png')
        plt.show()

        '''
        #PRINTS THE MATRIX IN READABLE FORM
        np.set_printoptions(threshold=np.inf)
        for cell in matrix:
            string = ''
            for item in cell:
                string = f'{string} {item.round(3)}'
            print(string)
        '''

    plt.figure()
    plt.xlim(min_x - 0.5, max_x + 0.5)
    plt.ylim(min_y - 0.5, max_y + 0.5)
    plt.scatter(x_list, y_list)

    plt.ylabel('Imaginary')
    plt.xlabel('Real')
    plt.savefig(f'eigens//eigens_{N}.png')
    plt.show()



        #print('')
        #print(f'eigens: {eigenvals}')
