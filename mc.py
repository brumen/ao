# specialized monte carlo for AirOptions
import numpy as np

from typing          import Tuple, List, Callable
from scipy.integrate import quad


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

    if drift_vol_ind == 'vol':  # vol integration
        return np.sqrt(quad( lambda x: sd_fct(ttm-(x-T_start))**2
                           , T_start
                           , T_end)[0] / (T_end - T_start))
    # drift integration
    return quad( lambda x: sd_fct(ttm-(x-T_start))
               , T_start
               , T_end)[0] / (T_end - T_start)


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


def mc_mult_steps( F_v     : [List, np.array]
                 , s_v     : np.array
                 , d_v     : np.array
                 , T_l     : np.array
                 , rho_m   : np.array
                 , T_v_exp: np.array
                 , nb_sim  = 1000
                 , model    = 'n'
                 , F_ret    = None
                 , keep_all_sims = False  ):  # -> [np.array, Dict[np.array]]:
    """ Multi-step monte-carlo integration of the ticket prices for one way Air options.

    :param F_v: list of forward values
    :param s_v: (vector of) volatility of the forward ticket process.
    :param d_v: drift of the forward ticket process.
    :param T_l: time points at which all F_v should be simulated
    :param rho_m: correlation matrix (2-dimensional array)
    :param T_v_exp: expiry of the forward contracts, maturity of aiplane tickets
    :param nb_sim: number of simulations
    :param model: model used for simulation, 'ln' for log-normal, or 'n' for normal
    :param F_ret: a sumulated list of return values - size (nb_sim, 1)
    :param keep_all_sims: keeps all simulations for each sim_times
    :returns: matrix of simulation values in the shape [simulation, fwd] if keep_all_sims = False,
              or dictionary, where keys are simulation times and values are simulations for those times.
    """

    one_step_model = ln_step if model == 'ln' else normal_step

    T_l_extend = add_zero_to_Tl(T_l)  # add 0 to simulation times if not already there
    T_l_diff   = np.diff(T_l_extend)
    nb_fwds    = len(F_v)

    F_sim = np.empty((nb_sim, nb_fwds))  # if not cuda_ind else gpa.empty((nb_sim, nb_fwds), np.double)
    F_sim[:, :] = F_v
    F_prev = F_sim  # previous simulations

    # also number of departure flights., nb_sim = nb. of simulations of those flights, also columns in F_sim
    nb_sim, nb_fwds = F_sim.shape  # nb of forward contracts, also nb of rows in F_sim

    for (T_diff, T_curr, T_prev) in zip(T_l_diff, T_l_extend[1:], T_l_extend[:-1]):  # T_diff = difference, T_curr: current time  # before: add_zero_to_Tl(T_l_diff)[:-1]
        ttm_used = np.array(T_v_exp) - T_curr
        s_v_used, d_v_used = vol_drift_vec(T_curr, T_diff, s_v, d_v, ttm_used)

        if nb_fwds == 1:
            rn_sim_l = np.random.normal(size=(1, nb_sim))
        else:
            rn_sim_l = np.random.multivariate_normal(np.zeros(nb_fwds), rho_m, size=nb_sim)  # if not cuda_ind else mn_gpu(0, rho_m, size=nb_sim)

        F_sim_next = one_step_model( F_sim if not keep_all_sims else F_prev
                                   , T_diff
                                   , s_v_used
                                   , d_v_used
                                   , rn_sim_l )

        if F_ret is None:  # no return flight given
            if not keep_all_sims:
                F_sim = np.maximum(F_sim_next, F_sim)
            else:
                yield (T_curr, np.maximum(F_sim_next, F_prev))
                F_prev = F_sim_next

        else:  # return flights
            F_sim_next_ret = F_sim_next + F_ret  # F_ret is already maximized over flights

            if not keep_all_sims:
                F_sim = np.maximum(F_sim_next_ret, F_sim)
            else:
                yield (T_curr, np.maximum(F_sim_next_ret, F_prev))
                F_prev = F_sim_next_ret

    if not keep_all_sims:
        yield (T_curr, F_sim)  # last value in T_curr


def add_zero_to_Tl(T_list : List[float]) -> np.array:
    """ Prepends a zero to sim_times if sim_times doesnt already have it.

    :param T_list:  list of simulation times
    :returns:       array with first element as 0, if it isnt already zero. potential zero added
    """

    if T_list[0] != 0.:
        T_l_local = np.zeros(len(T_list) + 1)
        T_l_local[1:] = T_list
        return T_l_local

    return T_list


def mc_mult_steps_ret( F_v     : Tuple[np.array, np.array]
                     , s_v     : Tuple[np.array, np.array]
                     , d_v     : Tuple[np.array, np.array]
                     , T_l     : Tuple[List[float], List[float]]
                     , rho_m   : Tuple[np.array, np.array]
                     , T_v_exp : Tuple[np.array, np.array]
                     , nb_sim        = 1000
                     , model         = 'n'
                     , keep_all_sims = False ):
    """ Simulates ticket prices for a return flight.

    :param F_v: tuple of list of forward values, first for departure, second for return (arrays in tuple are 1-dim)
    :param s_v: tuple of list of vols for tickets, first departure, then return
    :param d_v: drift of the process, drift
                  d_v[i] is a function of (ttm, params)
    :param T_l: list of time points at which all F_v should be simulated
    :param rho_m: correlation matrix (2-dim array)
    :param nb_sim: number of sims
    :param T_v_exp: expiry of forward contracts, expiry in numerical terms.
    :param model: model 'ln' or 'n' for normal
    :param cuda_ind: whether cuda or cpu is used (cuda = True, not cuda = False)
    :param keep_all_sims: whether to keep the simulations
    :returns: matrix [time_step, simulation, fwd] or ticket prices
    """

    F_v_dep,     F_v_ret     = F_v
    s_v_dep,     s_v_ret     = s_v
    d_v_dep,     d_v_ret     = d_v
    T_l_dep,     T_l_ret     = add_zero_to_Tl(T_l[0]), add_zero_to_Tl(T_l[1])
    T_v_exp_dep, T_v_exp_ret = T_v_exp  # expiry values 
    rho_m_dep,   rho_m_ret   = rho_m

    # mc_mult_steps is a generator and has exactly _0_ elements for keep_all_sims=False
    _, F_ret_realized = list(mc_mult_steps( F_v_ret
                                          , s_v_ret
                                          , d_v_ret
                                          , T_l_ret
                                          , rho_m_ret
                                          , T_v_exp_ret
                                          , nb_sim   = nb_sim
                                          , model    = model ))[0]

    return mc_mult_steps( F_v_dep
                        , s_v_dep
                        , d_v_dep
                        , T_l_dep
                        , rho_m_dep
                        , T_v_exp_dep
                        , nb_sim   = nb_sim
                        , model    = model
                        , F_ret    = np.amax( F_ret_realized, axis = 1 ).reshape((nb_sim, 1))  # simulations in columns
                        , keep_all_sims=keep_all_sims)
