# specialized monte carlo for AirOptions

import config
import sys
import numpy as np
import scipy.integrate
from numpy.random import multivariate_normal as mn_cpu
if sys.version_info < (3, 0):
    import cuda.vtpm_cpu as vtpm_cpu  # avx & omp analysis
else:
    import cuda.vtpm_cpu3 as vtpm_cpu

if config.CUDA_PRESENT:
    import cuda_ops as co
    import pycuda.gpuarray as gpa
    import curand
    rn_gen_global = curand.create_gen_simple()  # generator of random numbers
else:
    rn_gen_global = None


def integrate_fct( sd_fct
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
    log-normal step of one step monte carlo

    """

    if not cuda_ind:
        expon_part = (np.sqrt(T_diff) * s_v_used) * rn_sim_l - 0.5 * s_v_used ** 2 * T_diff
        expon_part += d_v_used * T_diff
        return F_sim_prev * np.exp(expon_part)
    else:  # gpu
        # F_sim_part = co.vtpm_cols_new_hd(s_v_used, rn_sim_l, tm_ind='t')
        # F_sim_part = co.vtpm_cols_new_hd(- 0.5 * s_v_used**2 * T_diff, F_sim_part, tm_ind='p')
        # F_sim_part = co.vtpm_cols_new_hd_ao(- 0.5 * s_v_used**2 * T_diff, s_v_used, rn_sim_l)
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
    :type s_v_used:    np.array, shape = (nb_contracts, 1)
    :param d_v_used:   drift of the contracts
    :type d_v_used:    np.array, shape = (nb_contracts, 1)
    :param rn_sim_l:   simulation numbers,
    :type rn_sim_l:    np.array, shape = (nb_contracts, nb_sims)
    :param cuda_ind:   indicator whether to use cuda or not
    :type cuda_ind:    bool
    """

    F_sim_next = np.empty_like(F_sim_prev)

    if not cuda_ind:
        # F_sim_next = F_sim_prev + s_v_used * np.sqrt(T_diff) * rn_sim_l
        # if d_v is not None:
        #    F_sim_next += d_v_used * T_diff
        F_sim_cols, F_sim_rows = F_sim_prev.shape
        vtpm_cpu.vm_ao( F_sim_prev
                      , d_v_used * T_diff
                      , s_v_used * np.sqrt(T_diff)
                      , rn_sim_l
                      , F_sim_next
                      , F_sim_cols
                      , F_sim_rows)
    else:
        F_sim_next = co.vtpm_cols_new_hd( s_v_used * np.sqrt(T_diff)
                                        , rn_sim_l
                                        , tm_ind = 't')
        F_sim_next = co.vtpm_cols_new_hd( d_v_used * T_diff
                                        , F_sim_next
                                        , tm_ind = 'p')
        F_sim_next += F_sim_prev

    return F_sim_next


def mc_mult_steps( F_v
                 , s_v
                 , d_v
                 , T_l
                 , rho_m
                 , nb_sim
                 , T_v_exp
                 , ao_f
                 , cva_vals = None
                 , model    = 'n'
                 , F_ret    = None
                 , cuda_ind = False):
    """
    Multi-step monte-carlo integration of the ticket prices

    :param F_v: list of forward values
    :type F_v:  list??/ numpy array of forward values 
    :param s_v: vector of vols for forward vals
    :type s_v:  np.array
    :param T_l: list of time points at which all F_v should be simulated
    :param rho_m: correlation matrix
    :param nb_sim: number of sims
    :param T_v_exp: expiry of the forward contracts, maturity of aiplane tickets
    :type T_v_exp:  np.array
    :param F_prev:  previous values of the forward process TODO
    :param ao_f: function to compute new vals from old ones
    :type ao_f:  TODO
    :param d_v: drift of the forward ticket process
    :type d_v:  np.array
    :param cva_vals: None ... everything is regenerated anew
                     a list of random numbers generated which can be reused all the time
    :param model: model used for simulation
    :type model:  string, 'ln' for log-normal, or 'n' for normal
    :param F_ret: a sumulated list of return values - size (nb_sim, 1)
    :param cuda_ind: cuda inidcator 
    :returns: matrix of simulation values in the shape [time_step, simulation, fwd] or new parameters
    :rtype:   TODO
    """

    if not cuda_ind:
        mn = mn_cpu
    else:
        mn = mn_gpu

    # add 0 to simulation times if not already there
    if T_l[0] != 0.:
        T_l_local = np.zeros(len(T_l)+1)
        T_l_local[1:] = T_l
    else:
        T_l_local = T_l

    nb_fwds = len(F_v)
    T_l_diff = np.diff(T_l_local)

    if not cuda_ind:
        F_sim_prev = np.empty((nb_fwds, nb_sim))
    else:
        F_sim_prev = gpa.empty( (nb_fwds, nb_sim)
                              , np.double)

    F_sim_l = None
    F_v_shape = F_v.shape

    # assumption F_v and s_v are in the same format
    if len(F_v_shape) == 1 or F_v_shape[0] == 1:
        F_v_used = F_v.reshape((nb_fwds, 1))  # F_v_used is column vector 
    else:
        F_v_used = F_v

    if not cuda_ind:
        F_sim_prev[:, :] = F_v_used
    else:  # write in F_sim_prev F_v_used by columns
        F_sim_prev = co.set_mat_by_vec(F_v_used, nb_sim)
        
    F_sim_next_new = F_sim_prev  # ao_p_next
        
    for T_ind, (T_diff, T_curr) in enumerate(zip(T_l_diff, T_l_local[:-1])):
        ttm_used = np.array(T_v_exp) - T_curr
        s_v_used = np.array([integrate_fct( lambda vol: s_vol
                                          , T_curr
                                          , T_curr + T_diff
                                          , ttm_curr
                                          , drift_vol_ind = 'vol')
                             for (s_vol, ttm_curr) in zip(s_v, ttm_used)])  # volatility depends on time-to-maturity

        d_v_used = np.array([integrate_fct( lambda drift: d_v_curr
                                          , T_curr
                                          , T_curr + T_diff
                                          , ttm_curr
                                          , drift_vol_ind = 'drift')
                             for (d_v_curr, ttm_curr) in zip(d_v, ttm_used)])

        if not cuda_ind:
            s_v_used = s_v_used.reshape((len(s_v_used), 1))
            d_v_used = d_v_used.reshape((len(d_v_used), 1))
        
        if nb_fwds == 1:
            rn_sim_l = np.random.normal(size=(nb_sim, 1))
        else:
            if cva_vals is None:
                if not cuda_ind:
                    rvs = mn(np.zeros(nb_fwds), rho_m, size=nb_sim).transpose()
                else:
                    rvs = mn_gpu(0, rho_m, size=nb_sim)
            else:
                rvs = cva_vals[T_ind]
            rn_sim_l = rvs

        one_step_params = [F_sim_prev, T_diff, s_v_used, d_v_used, rn_sim_l]
        if model == 'ln':  # log-normal model 
            one_step_model = ln_step
        else:  # normal model
            one_step_model = normal_step

        F_sim_next = one_step_model( *one_step_params
                                   , cuda_ind = cuda_ind)

        if F_ret is None:  # no return flight given
            F_sim_next_new = ao_f( F_sim_next, F_sim_next_new )
        else:
            if not cuda_ind:
                F_sim_next_used = F_sim_next + F_ret
            else:
                # F_dep_plus_ret = F_sim_next + F_ret
                # F_dep_plus_ret = co.vtpm_cols_new(F_ret, F_sim_next)
                # F_dep_plus_ret = co.vtpm_rows_new_ao(F_ret, F_sim_next)
                co.vtpm_rows_new_ao(F_ret, F_sim_next)  # WRONG WRONG WRONG WRONG WRONG WRONG
                F_sim_next_used = F_sim_next  # WRONG WRONG WRONG

            F_sim_next_new = ao_f(F_sim_next_used, F_sim_next_new)

        F_sim_prev = F_sim_next_new

    return F_sim_prev  # TODO: CHECK IF THIS IS CORRECT


def ao_compute_from_F( F_sim_l
                     , T_l
                     , ao_f
                     , ao_p
                     , cuda_ind = False):
    """
    computes the ao_f vectors from previously simulated F_sim_l

    :param F_sim_l:   list of simulated forward ticket prices
    :type F_sim_l:    numpy array of shape (len(T_l), nb_fwds, nb_sim))
    :param T_l:       list of time points at which all F_v should be simulated
    :param ao_f:      function to compute new vals from old ones
    :param ao_p:      initial parameters
    :param cuda_ind:  cuda inidcator 
    :returns:         ao_p_next
    :rtype:           ao_p_next # TODO: COMPLETE HERE??? 
    """

    ao_p_next = ao_p
        
    for T_ind in range(len(T_l) - 1): 
        F_sim_next = F_sim_l[T_ind+1, :, :]
        ao_p_next = ao_f(F_sim_next, ao_p_next, cuda_ind=cuda_ind)

    return ao_p_next


def add_zero_to_Tl(T_list):
    """
    adds a zero to T_l if T_list doesnt already have it

    :param T_list:  list of simulation times
    :type T_list:   list
    :returns:       array with potential zero added
    :rtype:         numpy array, with the first element 0
    """

    if T_list[0] != 0.:
        T_l_local = np.zeros(len(T_list) + 1)
        T_l_local[1:] = T_list
    else:
        T_l_local = T_list
    return T_l_local


def mc_mult_steps_ret( F_v
                     , s_v
                     , d_v
                     , T_l
                     , rho_m
                     , nb_sim
                     , T_v_exp
                     , ao_f
                     , ao_p      = None
                     , cva_vals  = None
                     , model     = 'n'
                     , cuda_ind  = False
                     , gen_first = True):
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
    :param ao_f: function to compute new vals from old ones
    :param ao_p: initial parameters
    :param d_v: drift of the process, drift
       d_v[i] is a function of (ttm, params)
    :param cva_vals: None ... everything is regenerated anew
                     a list of random numbers generated which can be reused all the time
    :param model:   model 'ln' or 'n' for normal
    :type model:    str
    :param cuda_ind: whether cuda or cpu is used
    :type cuda_ind:  bool
    :param gen_first: generates the return simulations first, then only runs over them 
    :returns:        matrix [time_step, simulation, fwd] or ticket prices
    """

    if not cuda_ind:
        mn = mn_cpu
    else:
        mn = mn_gpu

    F_v_dep, F_v_ret = F_v
    s_v_dep, s_v_ret = s_v
    d_v_dep, d_v_ret = d_v
    T_l_dep, T_l_ret = add_zero_to_Tl(T_l[0]), add_zero_to_Tl(T_l[1])
    T_v_exp_dep, T_v_exp_ret = T_v_exp  # expiry values 
    rho_m_dep, rho_m_ret = rho_m
    
    nb_fwds_dep, nb_fwd_ret = len(F_v_dep), len(F_v_ret)
    T_l_diff_dep = np.diff(T_l_dep)

    if not cuda_ind:
        F_sim_prev = np.empty((nb_fwds_dep, nb_sim))
        max_fct_used = np.maximum
    else:
        F_sim_prev = gpa.empty((nb_fwds_dep, nb_sim), np.double)
        max_fct_used = gpa.maximum

    F_v_shape = F_v_dep.shape

    # assumption F_v and s_v are in the same format
    if len(F_v_shape) == 1 or F_v_shape[0] == 1:
        F_v_used = F_v_dep.reshape((nb_fwds_dep, 1))
        T_v_used = T_v_exp_dep.reshape((nb_fwds_dep, 1))
    else:
        F_v_used = F_v_dep
        T_v_used = T_v_exp_dep

    if not cuda_ind:
        F_sim_prev[:, :] = F_v_used
    else:
        F_sim_prev = co.set_mat_by_vec(F_v_used, nb_sim)

    F_sim_next = np.empty((nb_fwds_dep, nb_sim))
        
    ao_p_next = ao_p

    if d_v is not None:
        d_v_ret_used = d_v_ret
    else:
        d_v_ret_used = None

    if gen_first:  # generates the return simulations & stores them
        F_sim_ret = mc_mult_steps( F_v_ret
                                 , s_v_ret
                                 , d_v_ret_used
                                 , T_l_ret
                                 , rho_m_ret
                                 , nb_sim
                                 , T_v_exp_ret
                                 , ao_f
                                 , cva_vals = cva_vals
                                 , model    = model
                                 , cuda_ind = cuda_ind)

        F_sim_ret_max = np.amax(np.amax(F_sim_ret, axis=1), axis=0)

    # iteration over simulation times 
    for T_ind, (T_diff, T_curr) in enumerate(zip(T_l_diff_dep, T_l_dep[:-1])):  # current time 
        ttm_dep = np.array(T_v_used) - T_curr  # time to maturity for departures 
        s_v_used = np.array([integrate_fct( lambda vol: s_vol
                                          , T_curr
                                          , T_curr + T_diff
                                          , ttm_curr
                                          , drift_vol_ind = 'vol')
                             for (s_vol, ttm_curr) in zip(s_v_dep, ttm_dep)])  # volatility depends on time-to-maturity
        if d_v is not None:
            d_v_used = np.array([integrate_fct( lambda drift: d_v_curr
                                              , T_curr
                                              , T_curr + T_diff
                                              , ttm_curr
                                              , drift_vol_ind = 'drift')
                                 for (d_v_curr, ttm_curr) in zip(d_v_dep, ttm_dep)])
            
        if not cuda_ind:
            s_v_used = s_v_used.reshape((len(s_v_used), 1))
            d_v_used = d_v_used.reshape((len(d_v_used), 1))
        
        if nb_fwds_dep == 1:
            rn_sim_l = np.random.normal(size=(nb_sim, 1))
        else:
            if cva_vals is None:
                if not cuda_ind:
                    rvs = mn(np.zeros(nb_fwds_dep), rho_m_dep, size = nb_sim).transpose()
                else:
                    rvs = mn_gpu(0, rho_m_dep, size = nb_sim)
            else:
                rvs = cva_vals[T_ind]
            rn_sim_l = rvs

        if model == 'ln':  # log-normal model 
            if not cuda_ind:
                # F_sim_next = F_sim_prev * np.exp(s_v_used * rn_sim_l - 0.5 * s_v_used**2 * T_diff)
                expon_part = (np.sqrt(T_diff) * s_v_used) * rn_sim_l - 0.5 * s_v_used**2 * T_diff
                if d_v is not None:
                    expon_part += d_v_used * T_diff
                F_sim_next = F_sim_prev * np.exp(expon_part)
            else:
                # F_sim_part = co.vtpm_cols_new_hd(s_v_used, rn_sim_l, tm_ind='t')
                # F_sim_part = co.vtpm_cols_new_hd(- 0.5 * s_v_used**2 * T_diff, F_sim_part, tm_ind='p')
                # F_sim_part = co.vtpm_cols_new_hd_ao(- 0.5 * s_v_used**2 * T_diff, s_v_used, rn_sim_l)
                co.vtpm_cols_new_hd_ao(- 0.5 * s_v_used**2 * T_diff, np.sqrt(T_diff) * s_v_used, rn_sim_l)
                F_sim_part = rn_sim_l
                if d_v is not None:
                    F_sim_part = co.vtpm_cols_new_hd(d_v_used * T_diff, F_sim_part, tm_ind='p')
                # F_sim_next = F_sim_prev * pycuda.cumath.exp(F_sim_part)
                co.sin_cos_exp_d( F_sim_part
                                , F_sim_part
                                , sin_cos_exp='exp')
                F_sim_next = F_sim_prev * F_sim_part
        else:  # normal model (not updated completely)
            if not cuda_ind:
                # these three lines below fully work
                # F_sim_next = F_sim_prev + s_v_used * np.sqrt(T_diff) * rn_sim_l
                # if d_v is not None:
                #     F_sim_next += d_v_used * T_diff
                vtpm_cpu.vm_ao(F_sim_prev, d_v_used * T_diff,
                               s_v_used * np.sqrt(T_diff), rn_sim_l, F_sim_next,
                               F_sim_prev.shape[0], F_sim_prev.shape[1])
            else:
                F_sim_next = co.vtpm_cols_new_hd(s_v_used * np.sqrt(T_diff), rn_sim_l, tm_ind='t')
                if d_v is not None:
                    F_sim_next = co.vtpm_cols_new_hd(d_v_used * T_diff, F_sim_next, tm_ind='p')
                F_sim_next += F_sim_prev

        # T_v_exp_ret_ch = T_v_exp_ret - T_curr
        if True:   # TODO: FIX THIS IN THE FUTURE ao_p['model'] == 'max':  # simplified model
            F_sim_next_max = np.amax(F_sim_next, axis = 0)
            F_res_simple = F_sim_next_max + F_sim_ret_max  # first return
            F_max_prev = np.maximum(F_res_simple, F_sim_prev)  # TODO: THIS HERE MIGHT BE WRONG WRONG WRONG

        else:
            for F_dep_idx in range(len(F_v_dep)):
                if not gen_first:
                    # F_sim_next are departure sims, F_sim_ret .. return sims
                    ao_p_tmp = mc_mult_steps( F_v_ret
                                            , s_v_ret
                                            , d_v_ret_used
                                            , T_l_ret
                                            , rho_m_ret
                                            , nb_sim
                                            , T_v_exp_ret
                                            , ao_f
                                            , cva_vals  = cva_vals
                                            # F_ret is simulations of that departure flight
                                            , F_ret     = F_sim_next[F_dep_idx, :]
                                            , cuda_ind  = cuda_ind)
                else:
                    ao_p_tmp = ao_compute_from_F( F_sim_ret + F_sim_next[F_dep_idx, :].reshape((1, 1, nb_sim)) # latter is vector (row)
                                                , T_l_ret
                                                , ao_f
                                                , ao_p
                                                , cuda_ind = False)

                if not cuda_ind:
                    # ao_p_next['F_max_prev'] = max_fct_used(ao_p_tmp['F_max_prev'],
                    #                                       ao_p_next['F_max_prev'])
                    vtpm_cpu.max2m( ao_p_tmp['F_max_prev']
                                  , ao_p_next['F_max_prev']
                                  , ao_p_next['F_max_prev']
                                  , ao_p_next['F_max_prev'].shape[0]
                                  , ao_p_next['F_max_prev'].shape[1])
                else:
                    ao_p_next['F_max_prev'] = max_fct_used( ao_p_tmp['F_max_prev']
                                                          , ao_p_next['F_max_prev'])
                    
        F_sim_prev = F_max_prev

    return F_sim_prev


def mn_gpu(unimp, rho_m, size=100):
    """
    standard multivariate normal with rho_m as correlation variable

    :param unimp: unimportant, so that it coincides w/ mn from numpy
    :type unimp:  None
    :param rho_m: correlation matrix
    :type rho_m:  rectangular 2-dimensional np.array
    :param size:  number of simulation variables
    :type size:   int
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
