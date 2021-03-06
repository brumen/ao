# Compute the nearest correlation matrix,
# as defined in a paper by
# Nicholas Higham  "Computing the Nearest Correlation matrix - A Problem in Finance"
#

import scipy
import scipy.linalg

from typing import Tuple
from numpy  import diag, eye, dot, sqrt, zeros, abs, array


def u_proj(A : array, W : array) -> array:
    """ Computes the projection of A to U projection

    """

    W_inv = scipy.linalg.inv(W)
    W_inv_had = W_inv * W_inv  # hadamard of W_inv
    theta = scipy.linalg.solve(W_inv_had, diag(A - eye(A.shape[0])))
    return A - dot(W_inv, dot(diag(theta), W_inv))


def mat_positive(A : array) -> array:
    """ Spectral decomposition of A, positive part of A.

    :param A: matrix of of which one wants the spectral decomposition of
    """

    A_eig_v, A_eig_m = scipy.linalg.eig(A)
    return dot(A_eig_m, dot(diag(0.5 * (A_eig_v + abs(A_eig_v))), A_eig_m.transpose()))


def mat_inv_sqrt(A : array) -> Tuple[array, array]:
    """
    computes the W^(0.5), W^(-0.5)

    :param A: matrix of of which one wants the spectral decomposition of
    :type A:  numpy matrix
    """

    U, S, VT = scipy.linalg.svd(A)
    res = dot(dot(U, diag(sqrt(S))), VT)

    return res, scipy.linalg.inv(res)


def s_proj(A : array, W : array) -> array:
    """ Computes S proj of A.
    """

    W_one_half, W_inv_one_half = mat_inv_sqrt(W)

    return dot(W_inv_one_half, dot(mat_positive(dot(W_one_half, dot(A, W_one_half))), W_inv_one_half))


def near_corr(A : array, W : array, nb_iter : int) -> array:
    """ Iterative algorithm for finding the nearest correlation matrix.

    """

    S = zeros(A.shape)
    Y = A

    for ind in range(nb_iter):
        R = Y - S
        X = s_proj(R, W)
        S = X - R
        Y = u_proj(X, W)

    return Y.real  # round away the zeros


def near_corr_simple(A : array, nb_iter=10) -> array:
    """
    computes the nearest correlation matrix to A

    :param A:  matrix to which the correlation is computed, a square matrix.
    """

    return near_corr(A, eye(A.shape[0]), nb_iter)
