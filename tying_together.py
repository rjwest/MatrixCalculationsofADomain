# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 19:14:43 2022

@author: Robert
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import time
import itertools
from multiprocessing.pool import Pool
import multiprocessing

from domain import Domain

# this computes the row of first variations of the length of the
# orbits with respect to the Fourier modes.  This method takes an
# optional flag for normalization; if it is set to True, the matrix
# rows will be normalized in such a way that would yield 1 on the
# diagonal for the disk

# Defines number of multithreaded pools **May require lower setting**
MAX_JOBS = multiprocessing.cpu_count()+1

def _build_matrix_row(Ω, N, q):
    # Set up uniform (equispaced) initial conditions for gradient
    # ascent Note that since the initial condition is symmetric
    # with respect to 0, so will be the solution.
    #start = time.time()

    Θ = Ω.maximal_marked_symmetric_orbit(q)

    row = _fetch_row_matrix_Anq(Ω,Θ,N)

    return row

def _fetch_row_matrix_Anq(Ω, Θ, N, normalize = False):
    # q is implied by the length of θ
    # Cache the Lazutkin coordinates of the orbit once and for all
    x=np.array([Ω.Lazutkin(θ) for θ in Θ])

    # Cache the angles φ_j
    Δ=np.diff(np.pad(Ω.γ(Θ),((0,0),(0,1)),'wrap'))

    sinφ = np.sin(np.arctan2(Δ[1],Δ[0])-Θ)

    if (normalize):
        normalization = 1./np.sum(sinφ)
        return [np.sum(np.cos(2*math.pi*k * x)*sinφ*normalization)
            for k in range(2, (N+2))]
    else:
        return [np.sum(np.cos(2*math.pi*k * x)*sinφ)
            for k in range(2, (N+2))]

# this computes all elements of the matrix.
def gen_matrix_Anq(Ω, N):
    matrix = np.array([[]])


    #Generate each index of a row in the matrix
    partial = time.time()

    #Create process pool, mapping to each row in the matrix to be built
    pool = Pool(MAX_JOBS)
    matrix = pool.starmap(_build_matrix_row, zip(itertools.repeat(Ω), itertools.repeat(N), range(2,(N+2))))
    pool.close()
    pool.join()

    end = time.time()

    print(f'{N} Rows computed in {end-partial}s')

    return np.array(matrix)

if __name__ == "__main__":

    #BUILD THE ORIGINAL DOMAIN
    Ω = Domain('coeff1.txt')

    N = 1000

    matrix = gen_matrix_Anq(Ω, N)

    temp = np.linalg.eigvals(matrix)
    x_list = [ele.real for ele in temp]
    y_list = [ele.imag for ele in temp]

    max_x = max(x_list)
    min_x = min(x_list)

    max_y = max(y_list)
    min_y = min(y_list)

    for n in range(2,N):
        inner_matrix = matrix[0:n,0:n]
        eigenvals = np.linalg.eigvals(inner_matrix)

        # extract real and imaginary
        x = [ele.real for ele in eigenvals]
        y = [ele.imag for ele in eigenvals]

        plt.figure()
        plt.xlim(min_x - 0.5, max_x + 0.5)
        plt.ylim(min_y - 0.5, max_y + 0.5)
        plt.scatter(x, y)

        plt.ylabel('Imaginary')
        plt.xlabel('Real')
        plt.savefig(f'eigens//eigens_{n:04}.png')
