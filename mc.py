# specialized monte carlo for AirOptions

import config
import numpy as np
import scipy.integrate
import logging

from numpy.random import multivariate_normal as mn_cpu

logger = logging.getLogger(__name__)

if config.CUDA_PRESENT:
    import cuda_ops as co
    import pycuda.gpuarray as gpa
    import curand
    rn_gen_global = curand.create_gen_simple()  # generator of random numbers
else:
    rn_gen_global = None


def integrate_vol_drift( sd_fct
                       , T_start
                       , T_end
                       , ttm
                       , drift_vol_ind = 'vol'):
    '''
    Integrates the drift/volatility of the diffusion process - used in option value calculation
    integrates the drift between the T_start and T_end, where T_start is ttm away from maturity

    :param sd_fct:        integrating function, for vol or drift
    :type sd_fct:         function of one argument
    :param T_start:       start of the integration
    :type T_start:        double
    :param T_end:         end of integration
    :type T_end:          double
    :param ttm:           time to maturity
    :type ttm:            double
    :param params:        parameters of the integrating fucntion???
    :type params:
    :param drift_vol_ind: indicator whether we are integrating drift or volatility
    :type drift_vol_ind:  string
    '''

    if drift_vol_ind == 'vol':  # vol integration
        return np.sqrt(scipy.integrate.quad( lambda x: sd_fct(ttm-(x-T_start))**2
                                           , T_start
                                           , T_end)[0] / (T_end - T_start))
    else:  # drift integration
        return scipy.integrate.quad( lambda x: sd_fct(ttm-(x-T_start))
                                   , T_start
                                   , T_end)[0] / (T_end - T_start)


def ln_step( F_sim_prev
           , T_diff
           , s_v_used
           , d_v_used
           , rn_sim_l
           , cuda_ind = False):
    """
    Log-normal step of one step monte carlo

    """

    if not cuda_ind:
        expon_part = (np.sqrt(T_diff) * s_v_used) * rn_sim_l - 0.5 * s_v_used ** 2 * T_diff
        expon_part += d_v_used * T_diff
        return F_sim_prev * np.exp(expon_part)
    else:  # gpu
        co.vtpm_cols_new_hd_ao(- 0.5 * s_v_used ** 2 * T_diff, np.sqrt(T_diff) * s_v_used, rn_sim_l)
        F_sim_part = rn_sim_l
        F_sim_part = co.vtpm_cols_new_hd(d_v_used * T_diff, F_sim_part, tm_ind='p')
        # F_sim_next = F_sim_prev * pycuda.cumath.exp(F_sim_part)
        co.sin_cos_exp_d(F_sim_part, F_sim_part, sin_cos_exp='exp')  # works faster than cumath
        # F_sim_next = F_sim_prev * pycuda.cumath.exp(F_sim_part)
        return F_sim_prev * F_sim_part


def normal_step( F_sim_prev
               , T_diff
               , s_v_used
               , d_v_used
               , rn_sim_l
               , cuda_ind = False):
    """
    one time step of the normal model

    :param F_sim_prev: previous simulated forward prices
    :type F_sim_prev:  np.array, shape: (nb_contracts, nb_sim)
    :param T_diff:     time difference between two simulation times
    :type T_diff:      double
    :param s_v_used:   volatilities
    :param d_v_used:   drift of the contracts
    :param rn_sim_l:   simulation numbers,
    :type rn_sim_l:    np.array, shape = (nb_contracts, nb_sims)
    :param cuda_ind:   indicator whether to use cuda or not, True or False
    """

    if not cuda_ind:
        F_sim_next = F_sim_prev + s_v_used * np.sqrt(T_diff) * rn_sim_l

        if d_v_used is not None:
           F_sim_next += d_v_used * T_diff

    else:
        F_sim_next = co.vtpm_cols_new_hd( s_v_used * np.sqrt(T_diff)
                                        , rn_sim_l
                                        , tm_ind = 't')
        F_sim_next = co.vtpm_cols_new_hd( d_v_used * T_diff
                                        , F_sim_next
                                        , tm_ind = 'p')
        F_sim_next += F_sim_prev

    return F_sim_next


def vol_drift_vec( T_curr : float
                 , T_diff : float
                 , s_v    : np.array
                 , d_v    : np.array
                 , ttm    : np.array
                 , integrate_fct = integrate_vol_drift ) -> np.array:
    """
    Creates the volatility and drift vector for the current time T_curr, until T_curr + T_diff
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


def mc_one_way( F_sim    : np.array
              , T_l_diff : np.array
              , T_v_exp  : np.array
              , s_v      : np.array
              , d_v      : np.array
              , rho_m    : np.array
              , model    ='normal'
              , F_ret    = None
              , cuda_ind = False ):
    """
    One way simulation of departure flights

    :param F_sim: initial flight values
    :type F_sim: 2-dimensional np.array(nb_simulations, nb_fwds)
    :param T_l_diff: difference between simulation times
    :type T_l_diff: list[double] or np.array
    :param T_v_exp: vector of departure dates (in numeric form)
    :param s_v: volatilities for departure (size = nb_departure flights)
    :param d_v: drift of departure flights (size = nb of departure flights)
    :param rho_m: correlation matrix
    :param model: underlying model: 'ln' or 'normal' - only 'normal' works so far
    :param F_ret: return flight, already maximized over the return flights
    :type F_ret: np.array(nb_simulations)
    :param cuda_ind: cuda indicator (True or False), by default False
    :type cuda_ind: bool
    """

    # also number of departure flights., nb_sim = nb. of simulations of those flights, also columns in F_sim
    nb_sim, nb_fwds = F_sim.shape  # nb of forward contracts, also nb of rows in F_sim

    T_l_local = add_zero_to_Tl(T_l_diff)  # TODO: THIS HERE IS PROBABLY WRONG

    for T_ind, (T_diff, T_curr) in enumerate(zip(T_l_diff, T_l_local[:-1])):
        ttm_used = np.array(T_v_exp) - T_curr
        s_v_used, d_v_used = vol_drift_vec(T_curr, T_diff, s_v, d_v, ttm_used)

        if nb_fwds == 1:
            rn_sim_l = np.random.normal(size=(1, nb_sim))
        else:
            if not cuda_ind:
                rn_sim_l = np.random.multivariate_normal(np.zeros(nb_fwds), rho_m, size=nb_sim)
            else:
                rn_sim_l = mn_gpu(0, rho_m, size=nb_sim)

        one_step_model  = ln_step if model == 'ln' else normal_step  # normal model
        F_sim_next      = one_step_model( F_sim
                                        , T_diff
                                        , s_v_used
                                        , d_v_used
                                        , rn_sim_l
                                        , cuda_ind = cuda_ind )

        if F_ret is None:  # no return flight given
            F_sim = np.maximum(F_sim_next, F_sim)  # more proper, although the same as w/ amax
        else:  # return flights
            if not cuda_ind:
                F_sim_next_ret = F_sim_next + F_ret  # F_ret is already maximized over flights
            else:
                # F_dep_plus_ret = F_sim_next + F_ret
                # F_dep_plus_ret = co.vtpm_cols_new(F_ret, F_sim_next)
                # F_dep_plus_ret = co.vtpm_rows_new_ao(F_ret, F_sim_next)
                co.vtpm_rows_new_ao(F_ret, F_sim_next)  # WRONG WRONG WRONG WRONG WRONG WRONG
                F_sim_next_ret = F_sim_next  # WRONG WRONG WRONG

            # F_sim_next_new = ao_f(F_sim_next_used, F_sim_next_new)
            F_sim = np.maximum(F_sim_next_ret, F_sim)

    return F_sim


def mc_mult_steps( F_v
                 , s_v     : np.array
                 , d_v     : np.array
                 , T_l     : np.array
                 , rho_m   : np.array
                 , nb_sim  : int
                 , T_v_exp : np.array
                 , model    = 'n'
                 , F_ret    = None
                 , cuda_ind = False):
    """
    Multi-step monte-carlo integration of the ticket prices for one way Air options

    :param F_v: list of forward values
    :type F_v:  list??/ numpy array of forward values 
    :param s_v: vector of vols for forward vals
    :param T_l: list of time points at which all F_v should be simulated
    :type T_l: np.array
    :param rho_m: correlation matrix
    :type rho_m: 2-dimensional np.array
    :param nb_sim: number of sims
    :param T_v_exp: expiry of the forward contracts, maturity of aiplane tickets
    :type T_v_exp:  np.array
    :param F_prev:  previous values of the forward process TODO
    :param d_v: drift of the forward ticket process
    :param model: model used for simulation, 'ln' for log-normal, or 'n' for normal
    :param F_ret: a sumulated list of return values - size (nb_sim, 1)
    :param cuda_ind: cuda indicator
    :returns: matrix of simulation values in the shape [time_step, simulation, fwd] or new parameters
    :rtype:   TODO
    """

    T_l_extend = add_zero_to_Tl(T_l)  # add 0 to simulation times if not already there
    nb_fwds = len(F_v)
    T_l_diff = np.diff(T_l_extend)

    F_sim = np.empty((nb_fwds, nb_sim)) if not cuda_ind else gpa.empty( (nb_fwds, nb_sim), np.double)

    # write F_v_used in F_sim_prev by columns
    if not cuda_ind:
        F_sim[:, :] = F_v
    else:
        F_sim = co.set_mat_by_vec(F_v, nb_sim)

    return mc_one_way( F_sim
                     , T_l_diff
                     , T_v_exp
                     , s_v
                     , d_v
                     , rho_m
                     , model = model
                     , F_ret = F_ret
                     , cuda_ind = cuda_ind)


def add_zero_to_Tl(T_list):
    """
    Adds a zero to T_l if T_list doesnt already have it.

    :param T_list:  list of simulation times
    :type T_list:   list
    :returns:       array with potential zero added
    :rtype:         numpy array, with the first element 0
    """

    if T_list[0] != 0.:
        T_l_local = np.zeros(len(T_list) + 1)
        T_l_local[1:] = T_list
        return T_l_local

    return T_list


def mc_mult_steps_ret( F_v
                     , s_v
                     , d_v
                     , T_l
                     , rho_m
                     , T_v_exp
                     , model     = 'n'
                     , cuda_ind  = False):
    """
    Simulates ticket prices for a return flight.

    :param F_v:     tuple of list of forward values, first for departure, second for return
    :type F_v:      tuple of 1-dimensional np.array
    :param s_v:     tuple of list of vols for tickets, first departure, then return
    :type s_v:      tuple of 1-dimensional np.array (np.array([]), np.array([]))
    :param T_l:     list of time points at which all F_v should be simulated
    :param rho_m:   correlation matrix
    :type rho_m:    2-dimensional np.array
    :param nb_sim:  number of sims
    :type nb_sim:   int
    :param T_v_exp: expiry of forward contracts
    :param d_v: drift of the process, drift
                  d_v[i] is a function of (ttm, params)
    :param model:   model 'ln' or 'n' for normal
    :type model:    str
    :param cuda_ind: whether cuda or cpu is used
    :type cuda_ind:  bool
    :returns:        matrix [time_step, simulation, fwd] or ticket prices
    """

    F_v_dep,     F_v_ret     = F_v
    s_v_dep,     s_v_ret     = s_v
    d_v_dep,     d_v_ret     = d_v
    T_l_dep,     T_l_ret     = add_zero_to_Tl(T_l[0]), add_zero_to_Tl(T_l[1])
    T_v_exp_dep, T_v_exp_ret = T_v_exp  # expiry values 
    rho_m_dep,   rho_m_ret   = rho_m

    return mc_one_way( F_v
                     , np.diff(T_l_dep)
                     , T_v_exp
                     , s_v_dep
                     , d_v_dep
                     , rho_m_dep
                     , cuda_ind = cuda_ind
                     , model    = model
                     , F_ret    = np.amax( mc_mult_steps( F_v_ret
                                                        , s_v_ret
                                                        , d_v_ret
                                                        , T_l_ret
                                                        , rho_m_ret
                                                        , T_v_exp_ret
                                                        , model    = model
                                                        , cuda_ind = cuda_ind )
                                         , axis = 0 )
                     )


def mn_gpu(unimp, rho_m, size=100):
    """
    Standard multivariate normal with rho_m as correlation variable

    :param unimp: unimportant, so that it coincides w/ mn from numpy
    :type unimp:  None
    :param rho_m: correlation matrix
    :type rho_m:  rectangular 2-dimensional np.array
    :param size:  number of simulation variables, an integer
    """

    nb_fwds = rho_m.shape[0]
    sim_rn_init = gpa.empty((nb_fwds, size), dtype=np.double)
    rho_m_chol_cuda = gpa.to_gpu(np.linalg.cholesky(rho_m))
    simulated_rn = gpa.empty((nb_fwds, size), dtype=np.double)
    curand.gen_eff_dev_rns_double( sim_rn_init.size
                                 , np.longlong(sim_rn_init.ptr)
                                 , rn_gen_global)
    co.matmul(rho_m_chol_cuda, sim_rn_init, simulated_rn)

    return simulated_rn
