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

# this computes the row of first variations of the length of the
# orbits with respect to the Fourier modes.  This method takes an
# optional flag for normalization; if it is set to True, the matrix
# rows will be normalized in such a way that would yield 1 on the
# diagonal for the disk

def _fetch_row_matrix_Anq(d, θ, N, normalize = False):
    # q is implied by the length of θ
    # Cache the Lazutkin coordinates of the orbit once and for all
    x=list(map(d.Lazutkin,θ))

    # Cache the angles φ_j
    p=list(map(d.γ,θ))
    q=len(θ)

    sinφ=[np.sin(np.arctan2(p[(j+1)%q][1] - p[j][1]
                            ,p[(j+1)%q][0] - p[j][0])-θ[j])
          for j in range(q)]

    if (normalize):
        normalization = 1/sum([ sinφ[j] for j in range(q)])
    else:
        normalization = 1;

    return [sum([ np.cos(2*math.pi*k * x[j])*sinφ[j]*normalization
                  for j in range(q)])
            for k in range(2, (N+2))]

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
        row = _fetch_row_matrix_Anq(d,Θ,N)
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
