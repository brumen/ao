# correlations/volatilities' module
import numpy as np


def corr_hyp_sec_basic(alpha, i, j):
    """
    correlation between months i, j

    :param alpha: cosh correlation parameter
    :type alpha:  double
    :param i:     first month
    :type i:      integer
    :param j:     second month
    :type j:      integer
    :returns:     correlation between months i, j
    :rtype:       double
    """
    return 1.0 / np.cosh(alpha * (i - j))  # TODO: This inversion can perhaps be written better


def corr_hyp_sec_two_fronts(rho, i, j):
    """
    converts the correlation parameter rho to a cosh parameter

    :param rho: correlation parameter
    :type rho:  double
    :param i:   first month
    :type i:    integer
    :param j:   second month
    :type j:    integer
    :returns:   correlation between the two months
    :rtype:     double
    """

    return corr_hyp_sec_basic(np.sqrt(2 * (1 - rho)), i, j)


def corr_hyp_sec_mat(rho, ind_range):
    """
    generates a correlation matrix from the hyp sec function above

    :param rho:       correlation parameter
    :type rho:        double
    :param ind_range: (row) vector of indices for each month the correlation is considered
    :type ind_range:  _row_ np.array
    :returns:         matrix of size (len(ind_range), len(ind_range)) for each months
    :rtype:           2-dimensional np.array
    """
    # return np.array([[corr_hyp_sec_two_fronts(rho, i, j)
    #                   for j in ind_range]
    #                  for i in ind_range])
    return corr_hyp_sec_two_fronts( rho
                                  , ind_range.reshape((len(ind_range), 1))
                                  , ind_range)
