# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 19:14:43 2022

@author: Robert
"""

import scipy.integrate as integrate
import math
import numpy as np
import matplotlib.pyplot as plt
import time

from gradient_descent_model import gradient_ascent
from tangent_parameterization import getPoints

from domain import Domain

# this returns the q-length function
def _l_of_thetas(x_list, y_list, q):
    summation = 0
    for j in range(q):
        summation += (math.sqrt((x_list[(j+1) % q] - x_list[j])**2 +
                                (y_list[(j+1) % q] - y_list[j])**2))
    return summation

# how is this different from the rho function in gradient_descend_model?
def _rho_of_theta(theta, pairs):
    bs = []
    for pair in pairs:
        a,b = pair.split(',')
        a = float(a)
        bs += [float(b)]
    summation = 0
    for k in range(len(pairs)):
        summation += bs[k] * np.cos(k*theta)
    return summation

# This returns the un-normalized Lazutkin parametrization at θ
def _z_of_theta(theta, pairs):
    return integrate.quad(lambda t,pairs: _rho_of_theta(t, pairs)**(1/3), 0, theta, args=(pairs))[0]

# This evaluates the k-th Fourier mode at theta
def  _e_n(theta, pairs, k, c):
    lp = (2*math.pi)/c

    return np.cos(k * (lp * _z_of_theta(theta, pairs)))

# this computes the first variation of the q-length with respect to e_k
def _fetch_val_matrix_Anq(gradient_thetas, k, c, q, pairs):
    summation = 0
    for j in range(q):
        # change names for x_list and y_list below
        x_list,y_list = getPoints(pairs, [gradient_thetas[j], gradient_thetas[(j + 1) % q]])

        alpha_l = np.arctan2(y_list[1] - y_list[0], x_list[1] - x_list[0])

#        if alpha_l < 0:
#            alpha_l += 2*math.pi

        plt.plot(x_list,y_list)

        #print(f'Argument of sine: {(alpha_l - gradient_thetas[j])/(2*math.pi)}\n j is: {j}\n q is: {q}')
        #print(f'alpha is: {alpha_l}\n Gradient theta is: {gradient_thetas[j]}')
        summation += _e_n(gradient_thetas[j], pairs, k, c) * np.sin(( alpha_l - gradient_thetas[j]))
    return summation

# this computes all elements of the matrix.
def gen_matrix_Anq(d, N):
    theta_list = []
    matrix = np.array([[]])
    non = 0
    c = _z_of_theta(2*math.pi, d.pairs)

    # each row is independent on the other rows; these can be computed in separate threads
    for q in range(2, (N+2), 1):
        # Set up uniform (equispaced) initial conditions for gradient
        # ascent Note that since the initial condition is symmetric
        # with respect to 0, so will be the solution.
        start = time.time()
        ep = 2*math.pi/q
        theta_list=[theta for theta in np.arange(0,(q*ep),ep)]

        gradient_thetas = gradient_ascent(d, theta_list, 0.01)

        #Generate each index of a row in the matrix
        partial = time.time()
        row = np.array([])
        for k in range(2, N+2):
            row = np.append(row, [_fetch_val_matrix_Anq(gradient_thetas, k, c, q, d.pairs)])
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
