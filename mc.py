# specialized monte carlo for AirOptions
import numpy as np

from typing          import Tuple, List, Callable
from scipy.integrate import quad
from functools       import lru_cache


def integrate_vol_drift( sd_fct  : Callable[[float], float]
                       , T_start : float
                       , T_end   : float
                       , ttm     : float
                       , drift_vol_ind = 'vol') -> float:
    """ Integrates the drift/volatility of the diffusion process - used in option value calculation
            integrates the drift between the T_start and T_end, where T_start is ttm away from maturity

    :param sd_fct: integrating function, for vol or drift, f(x) = vol/drift
    :param T_start: start of the integration
    :param T_end: end of integration
    :param ttm: time to maturity
    :param drift_vol_ind: indicator whether we are integrating drift or volatility ('vol' or 'drift')
    """

    integr_fct = lambda x: sd_fct(ttm-(x-T_start))

    if drift_vol_ind == 'vol':  # vol integration
        return np.sqrt(quad( lambda x: integr_fct(x)**2, T_start, T_end)[0] / (T_end - T_start))

    # drift integration
    return quad( integr_fct, T_start, T_end)[0] / (T_end - T_start)


def ln_step( F_sim_prev : np.array
           , T_diff     : np.array
           , s_v        : np.array
           , d_v        : np.array
           , rn_sim_l   : np.array ):
    """ Log-normal step of one step monte carlo.

    :param F_sim_prev: previous simulated forward prices, shape: (nb_contracts, nb_sim)
    :param T_diff: time difference between two simulation times
    :param s_v: volatility vector
    :param d_v: drift vector of the contracts
    :param rn_sim_l: simulation numbers, shape = (nb_contracts, nb_sims)
    """

    return F_sim_prev * np.exp((np.sqrt(T_diff) * s_v) * rn_sim_l - 0.5 * s_v ** 2 * T_diff + d_v * T_diff)


def normal_step( F_sim_prev  : np.array
               , T_diff      : float
               , s_v         : np.array
               , d_v         : [np.array, None]
               , rn_sim_l    : np.array ):
    """ One time step of the normal model

    :param F_sim_prev: previous simulated forward prices, shape: (nb_contracts, nb_sim)
    :param T_diff: time difference between two simulation times
    :param s_v: volatility vector
    :param d_v: drift vector of the contracts
    :param rn_sim_l: simulation numbers, shape = (nb_contracts, nb_sims)
    """

    F_sim_next = F_sim_prev + s_v * np.sqrt(T_diff) * rn_sim_l

    return F_sim_next + d_v * T_diff if d_v is not None else F_sim_next


def vol_drift_vec( T_curr : float
                 , T_diff : float
                 , s_v    : np.array
                 , d_v    : np.array
                 , ttm    : np.array
                 , integrate_fct = integrate_vol_drift ) -> Tuple[np.array, np.array]:
    """ Creates the volatility and drift vector for the current time T_curr, until T_curr + T_diff
           from s_v, d_v.

    :param T_curr: current time
    :param T_diff: difference to the next time
    :param s_v: array of vols for the corresponding flights
    :param d_v: array of drifts for the corresponding flights (could be none)
    :param ttm: to maturity for the corresponding flights
    :param integrate_fct: function that integrates from T_curr to T_curr + T_diff
                          the drift/volatility
    :returns: a pair of calculated volatilities/drifts for the flights
    """

    s_v_used = np.array([integrate_fct(lambda vol: s_vol
                                       , T_curr
                                       , T_curr + T_diff
                                       , ttm_curr
                                       , drift_vol_ind='vol')
                         for (s_vol, ttm_curr) in zip(s_v, ttm)])  # volatility depends on time-to-maturity

    if d_v is not None:
        d_v_used = np.array([integrate_fct(lambda drift: d_v_curr
                                          , T_curr
                                          , T_curr + T_diff
                                          , ttm_curr
                                          , drift_vol_ind='drift')
                             for (d_v_curr, ttm_curr) in zip(d_v, ttm)])

    else:
        d_v_used = None

    return s_v_used, d_v_used
