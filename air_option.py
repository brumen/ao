# air option computation file
import config
import datetime as dt
import numpy as np
import scipy.stats
import time
import multiprocessing as mp
import json
import mysql.connector
if config.CUDA_PRESENT:
    import cuda_ops
    import pycuda.gpuarray as gpa

import vols.vols as vols
import mc
import ds
import ao_codes
import air_search
import ao_params


DB_HOST  = 'odroid.local'
DATABASE = 'ao'
DB_USER  = 'brumen'


def air_option( F_v
              , s_v
              , T_l
              , K
              , p_c
              , d_c=None
              , nb_sim=1000
              , rho=0.9
              , model='max'
              , cuda_ind=False):
    """
    computes the value of the air option 

    :param F_v:        value of the forwards, air tickets considered
    :type F_v:         np.array 1-dimensional
    :param s_v:        volatility of air tickets
    :type s_v:         np.array 1- dimensional
    :param T_l:        list of dates on which tickets can be changed
    :type T_l:         list
    :param K:          strike of the option
    :type K:           double
    :param p_c:        probability of exercising
    :type p_c:         double
    :param d_c:        decreasing probability of exercising
    :type d_c:         double
    :param nb_sim:     number of simulations
    :type nb_sim:      integer
    :param rho:        correlation of simulations
    :type rho:         double
    :param model:      'max' or 'stat', for statistical
    :type model:       string
    :param cuda_ind:   indicator of the cuda
    :type cuda_ind:    bool
    """

    if type(F_v) is tuple:
        nb_fwds_dep, nb_fwds_ret = len(F_v[0]), len(F_v[1])
        nb_sim_times = len(T_l[0])  # TODO: THIS NEEDS UPDATING
        rho_m_dep, rho_m_ret = vols.corr_hyp_sec_mat(rho, range(nb_fwds_dep)), \
                               vols.corr_hyp_sec_mat(rho, range(nb_fwds_ret))
    else:  # only outgoing flight
        nb_fwds = len(F_v[0])
        nb_sim_times = len(T_l[0])  # TODO: THIS NEEDS UPDATING
        rho_m = vols.corr_hyp_sec_mat(rho, range(nb_fwds))  # TODO: THIS NEEDS TO BE UPDATED

    if model != 'max':
        d_v = vols.corr_hyp_sec_mat(d_c, range(nb_sim_times))[:, -1]
        d_v /= np.sum(d_v)
    else:
        d_v = None
    if not cuda_ind:
        F_sims = mc.mc_mult_steps_cpu(F_v, s_v, T_l, rho_m, nb_sim, d_v=d_v, model='n')
        max_f = np.maximum
    else:
        F_sims = mc.mc_mult_steps_cuda(F_v, s_v, T_l, rho_m, nb_sim, d_v=d_v, model='n')
        max_f = gpa.maximum
        
    # max over fwds
    F_max = np.amax(F_sims, axis=1)
    # max over times
    if model == 'max':
        F_max_max = np.amax(F_max, axis=0)
    else:
        F_max_max = F_max[0, :] * d_v[0]
        for st in range(1, nb_sim_times):
            F_max_max += F_max[st, :] * d_v[st]

    F_max_max_opt = F_max_max - K
    F_final = max_f(F_max_max_opt, 0.)
    if not cuda_ind:
        F_avg = np.mean(F_final)
    else:
        F_avg = np.mean(F_final.get())
    p_c_ch = 0.95 - p_c
    q95 = scipy.stats.mstats.mquantiles(F_max_max_opt, prob = p_c_ch)
    
    return {'avg': p_c * F_avg,
            'q95': q95[0]}


def air_option_seq( F_v
                  , s_v
                  , T_l
                  , T_mat
                  , K
                  , ao_f      = None
                  , ao_p      = None
                  , d_v       = None
                  , nb_sim    = 1000
                  , rho       = 0.9
                  , cuda_ind  = False
                  , underlyer ='n'
                  , gen_first = False):
    """
    computes the value of the air option, much lower memory impact, same as above

    :param F_v: vector of forward prices, or a tuple (F_1_v, F_2_v) for return flights
    :param s_v: vector of vols, or a tuple, similarly to F_v
    :param T_l: simulation list, same as s_v
    :param T_mat: maturity list 
    :param K: strike price
    :param ao_f: air_option function INSERT HERE
    :param ao_p: air_option parameters to be used in ao_f
    :param d_v: functions that describe the drift of the forward (list form)
       d_v[i] is a function of (F_prev, ttm, time_step, params)
    :param nb_sim: simulation number
    :param rho: correlation parameter, used only for now
    :param cuda_ind: indicator to use cuda
    :param underlyer: which model does the underlyer follow (normal 'n', log-normal 'ln')
    """

    return_flight_ind = type(F_v) is tuple

    if return_flight_ind:
        nb_fwds_dep, nb_fwds_ret = len(F_v[0]), len(F_v[1])
        rho_m_dep, rho_m_ret = vols.corr_hyp_sec_mat(rho, range(nb_fwds_dep)), \
                               vols.corr_hyp_sec_mat(rho, range(nb_fwds_ret))
        rho_m = (rho_m_dep, rho_m_ret)
    else:  # only outgoing flight
        nb_fwds = len(F_v)
        rho_m = vols.corr_hyp_sec_mat(rho, range(nb_fwds))

    if not return_flight_ind:
        mc_used = mc.mc_mult_steps_cpu
    else:
        mc_used = mc.mc_mult_steps_cpu_ret

    ao_final = mc_used( F_v
                      , s_v
                      , T_l
                      , rho_m
                      , nb_sim
                      , T_mat
                      , ao_f      = ao_f
                      , ao_p      = ao_p
                      , d_v       = d_v
                      , model     = underlyer
                      , cuda_ind  = cuda_ind
                      , gen_first = gen_first)

    if not cuda_ind:
        F_res = ao_final['F_max_prev']  # no averaging needed
    else:
        F_res = cuda_ops.amax_gpu_0(ao_final['F_max_prev'])

    F_max_max_opt = F_res - K

    if not cuda_ind:
        F_final = np.maximum(F_max_max_opt, 0.)
    else:
        F_final = gpa.maximum(F_max_max_opt, 0.)

    if not cuda_ind:
        F_avg = np.mean(F_final)
    else:
        F_avg = np.mean(F_final.get())

    return {'avg': F_avg,
            'q95': -1111.}


def ao_f( F_sims
        , ao_p
        , d_v    = None):
    """
    air_option in-between function, an example, others can be used

    :param ao_p['F_max_prev'] ... previous F_max
                                  has to return the next ao_p object
    """
    nb_tickets = F_sims.shape[0]
    # max over tickets
    if ao_p['model'] == 'max':
        F_max_max = np.maximum(F_sims, ao_p['F_max_prev'])
        ao_next = {'model': 'max',
                   'F_max_prev': F_max_max}
        return ao_next
    
    else:  # probability weighted model 
        F_new_max = np.sum(F_sims * d_v.reshape((nb_tickets, 1)), axis=0)
        F_max_max = np.maximum(F_new_max, ao_p['F_max_prev'])
        return {'model': 'prob',
                'F_max_prev': F_max_max}


def ao_f_arb( F_sims
            , ao_p
            , cuda_ind = False):
    """
    air_option in-between function avoiding the arbitrage condition 

    :param ao_p['F_max_prev'] ... previous F_max
    has to return the next ao_p object 
    """
    nb_tickets = F_sims.shape[0]
    # max over tickets
    if not cuda_ind:
        # the next 3 lines are correct
        # vtpm_cpu.max2m(F_sims, ao_p['F_max_prev'], F_max_max,
        #               F_max_max.shape[0],
        #               F_max_max.shape[1])
        # F_max_max = np.maximum(F_sims, ao_p['F_max_prev'])
        # arbitrage price of an option
        # F_max_arb = np.maximum(F_sims - (ao_p['K'] + ao_p['penalty']), 0.)
        # P_arg = np.mean(np.max(F_max_arb, axis=0))  # forwards are in columns
        F_max_max = np.maximum(np.amax(F_sims, axis=0), ao_p['F_max_prev'])
    else:
        F_max_max = gpa.maximum(F_sims, ao_p['F_max_prev'])

    ao_next = { 'model'     : 'max'
              , 'K'         : ao_p['K']
              , 'penalty'   : ao_p['penalty']
              , 'F_max_prev': F_max_max}
    return ao_next


def compute_option_raw( F_v
                      , s_v
                      , T_l_num
                      , T_mat_num
                      , K
                      , penalty
                      , p_c
                      , rho
                      , nb_sim    = 10000
                      , d_v       = None
                      , model     = 'max'
                      , cuda_ind  = False
                      , underlyer = 'n'
                      , gen_first = True):
    """
    computes the value of the option sequentially, in order to minimize memory footprint

    :param F_v:   vector of tickets
    """

    # sequential option parameter setup
    if type(F_v) is not tuple:
        F_v_len = len(F_v)
    else:
        F_v_len = len(F_v[1])  # iteration over return dates 
    if not cuda_ind:
        if model == 'max':  # lots of simplifications
            F_max_prev = np.zeros(nb_sim)
        else:
            F_max_prev = np.zeros((F_v_len, nb_sim))
    else:
        F_max_prev = gpa.zeros((F_v_len, nb_sim), dtype=np.double)
        
    ao_p = {'model': 'max',
            'F_max_prev': F_max_prev,
            'K': K,
            'penalty': penalty}

    opt_val_final = air_option_seq(F_v, s_v, T_l_num, T_mat_num,
                                   K, penalty, p_c,
                                   ao_f=ao_f_arb, ao_p=ao_p,
                                   d_v=d_v, nb_sim=nb_sim, rho=rho,
                                   model=model, cuda_ind=cuda_ind,
                                   underlyer=underlyer,
                                   gen_first=gen_first)

    # markups to the option value
    percentage_markup = ao_codes.reserves + ao_codes.tax_rate # 10 % markup 
    if type(F_v) is not tuple:
        F_v_max = np.max(F_v)
    else:
        F_v_max = max(np.max(F_v[0]), np.max(F_v[1]))
    min_payoff = max(50., F_v_max / ao_codes.ref_base_F * 50.)
    opt_val_final['avg'] = max(min_payoff, (1. + percentage_markup) * opt_val_final['avg'])
    return opt_val_final
    

def construct_st(date_s, date_e, date_t_dt, simplify_compute):
    """
    used for construction of simulation times 
    :param ate_s: date start
    :param date_e: date end
    :param date_t_dt: date today (in dt format)
    """
    T_l = ds.construct_date_range(date_s, date_e)  # in date format
    if simplify_compute == 'all_sim_dates':
        T_l_num = [(date_sim - date_t_dt).days/365. for date_sim in T_l]
    elif simplify_compute == 'take_last_only':
        T_l_num = [(T_l[-1] - date_t_dt).days/365.]

    return T_l_num


def find_minmax_ow(rof):
    """
    does that for one-way flights; adds fields 'min_max' to reorg_flights_v

    :param rof: TODO HERE
    :type rof:  TODO HERE
    """
    min_max_dict = dict()  # new dict to return
    change_dates = rof.keys()
    total_min, total_max = 1000000., 0.
    for c_date in change_dates:
        min_max_dict[c_date] = dict()
        flights_by_daytime = rof[c_date]
        flight_daytimes = rof[c_date].keys()
        cd_min, cd_max = 1000000., 0.
        for f_daytime in flight_daytimes:
            flight_subset = flights_by_daytime[f_daytime]
            # now find minimum or maximum
            min_subset, max_subset = 1000000., 0.  # one million
            for d_date in flight_subset:
                if flight_subset[d_date][5] < min_subset:
                    min_subset = flight_subset[d_date][5]
                if flight_subset[d_date][5] >= max_subset:
                    max_subset = flight_subset[d_date][5]
            flight_subset['min_max'] = (min_subset, max_subset)
            min_max_dict[c_date][f_daytime] = (min_subset, max_subset)
            if min_subset < cd_min:
                cd_min = min_subset
            if max_subset >= cd_max:
                cd_max = max_subset
        min_max_dict[c_date]['min_max'] = (cd_min, cd_max)
        if total_min > cd_min:
            total_min = cd_min
        if total_max < cd_max:
            total_max = cd_max
    min_max_dict['min_max'] = (total_min, total_max)
    return min_max_dict


def find_minmax_flight_subset(reorg_flights_v, ret_ind=False):
    """
    finds the minimum and maximum of flights in each subset 
    :param reorg_flights_v: dictionary structure of flights 
    :param ret_ind: indicator of return flight 
    :return min_max subset 
    """

    if not ret_ind:  # outbound flight only
        return find_minmax_ow(reorg_flights_v)
    else:  # return flight 
        return find_minmax_ow(reorg_flights_v[0]), find_minmax_ow(reorg_flights_v[1])

    
# these two functions (d_v_fct and s_v_fct) are here for pickle reasons
def s_v_fct(s, t):
    """
    volatility structure of the model,

    :param s: volatility at time t
    :type s:  double
    :param t: time at which volatility is evaluted
    :type t:  double
    :returns: volatility of the model
    :rtype:   double
    """
    return s

    
def d_v_fct(d, t):

    return d


def construct_dr(date_s, date_e):
    """
    constructs the interval for option extension

    :param date_s:  start date
    :type date_s:   string in the format '2017-02-05'
    :param date_e:  end date
    :type date_e:   string in the format '2017-02-05
    :returns:
    """
    y_b, m_b, d_b = date_s.split('-')
    outbound_date_start_nf = y_b + m_b + d_b
    y_e, m_e, d_e = date_e.split('-')
    outbound_date_end_nf = y_e + m_e + d_e
    outbound_date_range = ds.construct_date_range(outbound_date_start_nf, outbound_date_end_nf)
    outbound_date_range_minus = [ds.convert_dt_minus(x) for x in outbound_date_range]

    return outbound_date_range_minus


def obtain_flights_mat( flights
                      , flights_include
                      , date_today_dt):
    """
    # constucting the flight maturity
    # flights in the form [(id, dep, arr, price, flight_nb)...]
    # flights include: dict of flights as in reorg_flights,
    # censor flights that dont belong

    """
    flights_mat = []
    for dd in flights:
        dd_day, dd_time = dd[1].split('T')
        dd_tod = ao_codes.get_tod(dd_time)
        flight_mat_res = (ds.convert_datedash_date(dd_day) - date_today_dt).days / 365.
        if flights_include is None:
            flights_mat.append(flight_mat_res)
        else:
            if flights_include[dd_day][dd_tod][dd_time][-1]:
                flights_mat.append(flight_mat_res)
    return flights_mat


def sort_all(F_v, F_mat, s_v, d_v, fl_v):
    """
    sorts the flights according to the F_v, assmption being that similar flights are most correlated

    :param F_v:
    """

    zip_ls = sorted(zip(F_v, F_mat, s_v, d_v, fl_v))
    F_v_s, F_mat_s, s_v_s, d_v_s, fl_v_s = zip(*zip_ls)
    return F_v_s, F_mat_s, s_v_s, d_v_s, fl_v_s


def obtain_flights(io_dr_minus, flights_include, io_ind='out',
                   correct_drift=True,
                   write_data_progress=None,
                   is_return_for_writing=True):
    """
    # get the flights for outbound and inbound flight
    # io_dr_minus: input/output date range _minus (with - sign)
    # io_ind: inbound/outbound indicator

    """

    F_v, flights_v, F_mat, s_v_obtain, d_v_obtain = [], [], [], [], []
    reorg_flights_v = dict()
    if io_ind == 'out':  # outbound
        origin_used, dest_used = origin_place, dest_place
    else:
        origin_used, dest_used = dest_place, origin_place

    mysql_conn_gtp = mysql.connector.connect( host     = DB_HOST
                                            , database = DATABASE
                                            , user     = DB_USER
                                            , password = ao_codes.brumen_mysql_pass)

    for od in io_dr_minus:
        if write_data_progress is not None:  # write progress into file
            fo = open(write_data_progress, 'w')
            fo.write(json.dumps({'is_complete': False,
                                 'progress_notice': 'Fetching flights for ' + str(od)}))
            fo.close()

        ticket_val, flights, reorg_flights = \
            air_search.get_ticket_prices(origin_place       = origin_used,
                                         dest_place         = dest_used,
                                         outbound_date      = od,
                                         country            = country,
                                         currency           = currency,
                                         locale             = locale,
                                         include_carriers   = carrier,
                                         cabinclass         = cabinclass,
                                         adults             = adults,
                                         errors             = errors,
                                         insert_into_livedb = insert_into_livedb,
                                         use_mysql_conn     = mysql_conn_gtp)

        if write_data_progress is not None:  # write progress into file
            last_elt = od == io_dr_minus[-1]
            fo = open(write_data_progress, 'w')
            fo.write(json.dumps({'is_complete': False,  # is_return_for_writing and last_elt,
                                 'progress_notice': 'Fetched flights for ' + str(od)}))
            fo.close()

        # does the flight exist for that date??
        if reorg_flights.has_key(od):
            F_v.extend(ticket_val)
            flight_dep_time_added = [x[1] for x in flights]  # just the departure time
            io_dr_drift_vol = ao_params.get_drift_vol_from_db_precise(od,
                                                                      flight_dep_time_added,
                                                                      orig=origin_used,
                                                                      dest=dest_used,
                                                                      carrier=carrier,
                                                                      correct_drift=correct_drift,
                                                                      fwd_value=np.mean(ticket_val))
            io_dr_vol = [x[0] for x in io_dr_drift_vol]
            io_dr_drift = [x[1] for x in io_dr_drift_vol]
            s_v_obtain.extend(io_dr_vol)  # adding the vols
            d_v_obtain.extend(io_dr_drift)  # adding the drifts
            flights_v.extend(flights)
            F_mat.extend(obtain_flights_mat(flights, flights_include, date_today_dt))  # maturity of forwards
            reorg_flights_v[od] = reorg_flights[od]

    F_v = np.array(F_v)
    F_mat = np.array(F_mat)
    if len(F_v) > 0:  # there are actual flights
        return F_v, F_mat, s_v_obtain, d_v_obtain, flights_v, reorg_flights_v, 'Valid'
    else:  # no flights, indicate that it is wrong
        return [], [], [], [], [], [], 'Invalid'


def filter_prices_and_flights(price_l, flights_l, reorg_flights_l, flights_include):
    """
    fliter prices from flights_include

    """

    F_v, flight_v = [], []
    reorg_flight_v = {}
    for flight_p, flight_info in zip(price_l, flights_l):
        if flights_include is None:  # include all flights
            F_v.append(flight_p)
            flight_v.append(flight_info)
        else:  # include only selected flights
            flight_date, flight_hour = flight_info[1].split('T')
            flight_tod = ao_codes.get_tod(flight_hour)
            # check if the flight in flights_include exists in flights_l (no shananigans)
            fcc = (flight_date in flights_include.keys()) and (flight_tod in flights_include[flight_date]) \
                  and (flight_hour in flights_include[flight_date][flight_tod])

            if fcc:
                if flights_include[flight_date][flight_tod][flight_hour][-1]:
                    F_v.append(flight_p)
                    flight_v.append(flight_info)
            else:  # shananigan happening
                return [], [], [], 'Invalid'

    # reconstructed reorg_flights_v ??? WHY IS THIS NECESSARY
    for time_of_day in reorg_flights_l:
        reorg_flight_v[time_of_day] = {}
        tod_flights = reorg_flights_l[time_of_day]
        for dep_time in tod_flights:
            if flights_include is None:
                reorg_flight_v[time_of_day][dep_time] = reorg_flights_l[time_of_day][dep_time]
            else:
                if reorg_flights_l[time_of_day][dep_time][6] in flights_include:
                    reorg_flight_v[time_of_day][dep_time] = reorg_flights_l[time_of_day][dep_time]

    return F_v, flight_v, reorg_flight_v, 'Valid'


def obtain_flights_recompute(io_dr_minus, flights_include,
                             io_ind='out', correct_drift=True,
                             write_data_progress=None,
                             is_return_for_writing=True):
    """
    # io_dr_minus: input/output date range _minus (with - sign)
    # io_ind: inbound/outbound indicator
    # all flights are in flights_include
    # F_mat ... maturity of the forward prices

    """

    F_v, flights_v, F_mat, s_v_obtain, d_v_obtain = [], [], [], [], []
    reorg_flights_v = dict()
    if io_ind == 'out':  # outbound
        origin_used, dest_used = origin_place, dest_place
    else:
        origin_used, dest_used = dest_place, origin_place

    for od in io_dr_minus:
        # fliter prices from flights_include
        ticket_val = []
        flights = []  # (id, dep, arr, price, flight_id)
        reorg_flight = {}
        for tod in flights_include[od]:  # iterating over time of day
            reorg_flight[tod] = {}
            for dep_time in flights_include[od][tod]:
                res = flights_include[od][tod][dep_time]
                if dep_time != 'min_max':
                    flight_id, _, dep_time, arr_date, arr_time, flight_price, flight_id, flight_included = res
                    carrier = flight_id[:2]  # first two letters of id - somewhat redundant
                    if flight_included:
                        ticket_val.append(flight_price)
                        flights.append((flight_id,
                                        od + 'T' + dep_time,
                                        arr_date + 'T' + arr_time,
                                        flight_price,
                                        flight_id))
                        reorg_flight[tod][dep_time] = flights_include[od][tod][dep_time]

        # add together
        F_v.extend(ticket_val)
        flight_dep_time_added = [x[1] for x in flights]  # just the departure time
        io_dr_drift_vol = ao_params.get_drift_vol_from_db_precise( od
                                                                 , flight_dep_time_added
                                                                 , orig          = origin_used
                                                                 , dest          = dest_used
                                                                 , carrier       = carrier
                                                                 , correct_drift = correct_drift
                                                                 , fwd_value     = np.mean(ticket_val))
        io_dr_vol = [x[0] for x in io_dr_drift_vol]
        io_dr_drift = [x[1] for x in io_dr_drift_vol]
        s_v_obtain.extend(io_dr_vol)  # adding the vols
        d_v_obtain.extend(io_dr_drift)  # adding the drifts
        flights_v.extend(flights)
        F_mat.extend(obtain_flights_mat(flights, flights_include, date_today_dt))  # maturity of forwards
        reorg_flights_v[od] = reorg_flight

    F_v = np.array(F_v)
    F_mat = np.array(F_mat)
    return F_v, F_mat, s_v_obtain, d_v_obtain, flights_v, reorg_flights_v, 'Valid'


def get_flight_data( flights_include     = None
                   , origin_place        = 'SFO'
                   , dest_place          = 'EWR'
                   , outbound_date_start = '2017-02-25'
                   , outbound_date_end   = '2017-02-26'
                   , inbound_date_start  = '2017-03-12'
                   , inbound_date_end    = '2017-03-13'
                   , carrier             = 'UA'
                   , country             = 'US'
                   , currency            = 'USD'
                   , locale              = 'en-US'
                   , cabinclass          = 'Economy'
                   , adults              = 1
                   , errors              = 'graceful'
                   , mt_ind              = True
                   , return_flight       = False
                   , recompute_ind       = False
                   , correct_drift       = True
                   , insert_into_livedb  = True
                   , write_data_progress = None):
    """
    get flight data, no computing 
    :param flights_include: if None - include all, otherwise remove the flights 
                            not in flights_include 
    :param write_data_progress: write progress of fetching data into the filename given 
    """
    # constuct simulation times
    lt = time.localtime()
    date_today = str(lt.tm_year) + str(ds.d2s(lt.tm_mon)) + str(ds.d2s(lt.tm_mday))
    date_today_dt = ds.convert_str_date(date_today)

    out_dr_minus = construct_dr(outbound_date_start, outbound_date_end)
    if return_flight:
        in_dr_minus = construct_dr(inbound_date_start, inbound_date_end)

    if recompute_ind:
        obtain_flights_f = obtain_flights_recompute
    else:
        obtain_flights_f = obtain_flights
        
    # departure flights, always establish
    if not return_flight:
        F_v_dep_uns, F_mat_dep_uns, s_v_dep_u_uns, d_v_dep_u_uns, \
            flights_v_dep_uns, reorg_flights_v_dep, valid_check = obtain_flights_f(out_dr_minus, flights_include, io_ind='out',
                                                                                   correct_drift=correct_drift,
                                                                                   write_data_progress=write_data_progress,
                                                                                   is_return_for_writing=True)
        if valid_check != 'Valid':  # not valid, return immediately
            return [], [], [], [], [], [], False
        
        F_v_dep, F_mat_dep, s_v_dep_u, d_v_dep_u, \
            flights_v_dep = sort_all(F_v_dep_uns, F_mat_dep_uns, s_v_dep_u_uns, d_v_dep_u_uns,
                                     flights_v_dep_uns)
        F_v_dep = np.array(F_v_dep)  # these are np.arrays, correct back 
        F_mat_dep = np.array(F_mat_dep)
        if valid_check == 'Valid':
            s_v_dep = [lambda t: s_v_fct(s_v_u, t) for s_v_u in s_v_dep_u]
            d_v_dep = [lambda t: d_v_fct(d_v_u, t) for d_v_u in d_v_dep_u]
    else:
        if flights_include is None:
            F_v_dep_uns, F_mat_dep_uns, s_v_dep_raw_uns, d_v_dep_raw_uns, \
                flights_v_dep_uns, reorg_flights_v_dep, valid_check_out = \
                    obtain_flights_f(out_dr_minus, 
                                     flights_include, io_ind='out', correct_drift=correct_drift,
                                     write_data_progress=write_data_progress,
                                     is_return_for_writing=True)
            if valid_check_out != 'Valid':  # not valid, return immediately
                return ([], []), ([], []),  ([], []), ([], []), ([], []), ([], []), False


            F_v_dep, F_mat_dep, s_v_dep_raw, d_v_dep_raw, \
                flights_v_dep = sort_all(F_v_dep_uns, F_mat_dep_uns, s_v_dep_raw_uns, d_v_dep_raw_uns,
                                                              flights_v_dep_uns)
            F_v_dep = np.array(F_v_dep)  # these are np.arrays, correct back 
            F_mat_dep = np.array(F_mat_dep)
            
            F_v_ret_uns, F_mat_ret_uns, s_v_ret_raw_uns, d_v_ret_raw_uns, \
                flights_v_ret_uns, reorg_flights_v_ret, valid_check_in = obtain_flights_f(in_dr_minus, 
                                                                                          flights_include, io_ind='in', correct_drift=correct_drift,
                                                                                          write_data_progress=write_data_progress,
                                                                                          is_return_for_writing=True)
            if valid_check_in != 'Valid':  # not valid, return immediately
                return ([], []), ([], []),  ([], []), ([], []), ([], []), ([], []), False

            F_v_ret, F_mat_ret, s_v_ret_raw, d_v_ret_raw, \
                flights_v_ret = sort_all(F_v_ret_uns, F_mat_ret_uns, s_v_ret_raw_uns, d_v_ret_raw_uns, \
                                                              flights_v_ret_uns)
            F_v_ret = np.array(F_v_ret)  # these are np.arrays, correct back 
            F_mat_ret = np.array(F_mat_ret)
            
        else:
            F_v_dep_uns, F_mat_dep_uns, s_v_dep_raw_uns, d_v_dep_raw_uns, \
                flights_v_dep_uns, reorg_flights_v_dep, valid_check_out = obtain_flights_f(out_dr_minus,
                                                                                           flights_include[0], io_ind='out', correct_drift=correct_drift,
                                                                                           write_data_progress=write_data_progress,
                                                                                           is_return_for_writing=True)
            if valid_check_out != 'Valid':  # not valid, return immediately
                return ([], []), ([], []),  ([], []), ([], []), ([], []), ([], []), False

            F_v_dep, F_mat_dep, s_v_dep_raw, d_v_dep_raw, \
                flights_v_dep = sort_all(F_v_dep_uns, F_mat_dep_uns, s_v_dep_raw_uns, d_v_dep_raw_uns, \
                                         flights_v_dep_uns)
            F_v_dep = np.array(F_v_dep)  # these are np.arrays, correct back 
            F_mat_dep = np.array(F_mat_dep)
            
            F_v_ret_uns, F_mat_ret_uns, s_v_ret_raw_uns, d_v_ret_raw_uns, \
                flights_v_ret_uns, reorg_flights_v_ret, valid_check_in = obtain_flights_f(in_dr_minus,
                                                                                          flights_include[1], io_ind='in', correct_drift=correct_drift,
                                                                                          write_data_progress=write_data_progress,
                                                                                          is_return_for_writing=True)
            if valid_check_in != 'Valid':  # not valid, return immediately
                return ([], []), ([], []),  ([], []), ([], []), ([], []), ([], []), False

            F_v_ret, F_mat_ret, s_v_ret_raw, d_v_ret_raw, \
                flights_v_ret = sort_all(F_v_ret_uns, F_mat_ret_uns, s_v_ret_raw_uns, d_v_ret_raw_uns,
                                         flights_v_ret_uns)
            F_v_ret = np.array(F_v_ret)  # these are np.arrays, correct back 
            F_mat_ret = np.array(F_mat_ret)
            
        valid_check = (valid_check_out == 'Valid') and (valid_check_in == 'Valid')
        if valid_check:
            s_v_dep = [lambda t: s_v_fct(s_elt, t) for s_elt in s_v_dep_raw]
            d_v_dep = [lambda t: d_v_fct(d_elt, t) for d_elt in d_v_dep_raw]
            s_v_ret = [lambda t: s_v_fct(s_elt, t) for s_elt in s_v_ret_raw]
            d_v_ret = [lambda t: d_v_fct(d_elt, t) for d_elt in d_v_ret_raw]

    if valid_check:
        if not return_flight:
            return F_v_dep, F_mat_dep, flights_v_dep, reorg_flights_v_dep, s_v_dep, d_v_dep, True
        else:
            return (F_v_dep, F_v_ret), (F_mat_dep, F_mat_ret), \
                (flights_v_dep, flights_v_ret), (reorg_flights_v_dep, reorg_flights_v_ret), \
                (s_v_dep, s_v_ret), (d_v_dep, d_v_ret), True
    else:  # not valid
        if not return_flight:
            return [], [], [], [], [], [], False
        else:
            return ([], []), ([], []), ([], []), ([], []), ([], []), ([], []), False


def compute_date_by_fraction(dt_today, dt_final, fract, total_fraction):
    """

    :param dt_today:       "today's" date in datetime.date format
    :type dt_today:        datetime.date
    :param dt_final:       final date that one considers for excersing the option
    :type dt_final:        datetime.date
    :param fract:          the fraction of the days between dt_today and dt_final (usually 3)
    :type fract:           integer
    :param total_fraction: total number of options that one considers (usually 3)
    :type total_fraction:  integer
    :returns:              outbound date fract/total_fraction between dt_today and dt_final
    :rtype:                datetime.date
    """

    # fraction needs to be an integer
    outbound_dt = ds.convert_datedash_date(dt_final)
    # - 3 ... no change in the last 3 days
    outbound_day_diff = (outbound_dt - dt_today).days * fract/total_fraction - 3  # integer
    outbound_date_consid = dt_today + dt.timedelta(days=outbound_day_diff)
    outbound_date_consid = ds.convert_datetime_str(outbound_date_consid)
    return outbound_date_consid


def compute_option_val( origin_place          = 'SFO'
                      , dest_place            = 'EWR'
                      , flights_include       = None
                       # when can you change the option 
                      , option_start_date     = '20170522'
                      , option_end_date       = '20170524'
                      , option_ret_start_date ='20170601'
                      , option_ret_end_date   ='20170603',
                       # next 4 - when do the (changed) flights occur
                       outbound_date_start='2017-05-25',
                       outbound_date_end='2017-05-26',
                       inbound_date_start='2017-06-05',
                       inbound_date_end='2017-06-06',
                       K=1600.,  # option strike price (return or combined)
                       penalty=100.,  # if return, this is counted 2x
                       p_c=0.2,  # probability of changing
                       carrier='UA',
                       nb_sim=10000,  # CHECK THIS 
                       rho=0.95,
                       country='US', currency='USD', locale='en-US',
                       adults=1,
                       cabinclass='Economy', 
                       model='max', cuda_ind=False,
                       debug=False,
                       errors='graceful',
                       simplify_compute='take_last_only',  # 'all_sim_dates'
                       # simplify_compute='all_sim_dates',  # 'all_sim_dates'
                       underlyer='n',
                       mt_ind=False,
                       price_by_range=True,
                       return_flight=False,
                       res_supplied=None,
                       gen_first=True,
                       recompute_ind=False,
                       correct_drift=True,
                       write_data_progress=None):
    """
    computes the option value by getting data from skyscanner
    :param simplify_compute: simplifies the computation in that it only simulates the last simulation date
       options are: "take_last_only", "all_sim_dates"
    :param write_data_progress: if None, just compute the option 
                                if filename given, then write into that filename the progress 
    """
    # date today 
    lt = time.localtime()
    date_today = str(lt.tm_year) + str(ds.d2s(lt.tm_mon)) + str(ds.d2s(lt.tm_mday))
    date_today_dt = ds.convert_str_date(date_today)
    if res_supplied is None:
        res = get_flight_data(flights_include=flights_include,
                              origin_place=origin_place,
                              dest_place=dest_place,
                              # when can you change the option 
                              option_start_date=option_start_date,
                              option_end_date=option_end_date,
                              option_ret_start_date=option_ret_start_date,
                              option_ret_end_date=option_ret_end_date,
                              # next 4 - when do the (changed) flights occur
                              outbound_date_start=outbound_date_start,
                              outbound_date_end=outbound_date_end,
                              inbound_date_start=inbound_date_start,
                              inbound_date_end=inbound_date_end,
                              K=K,  # option strike price (return or combined)
                              p_c=p_c,  # probability of changing
                              carrier=carrier,
                              nb_sim=nb_sim,
                              rho=rho,
                              country=country, currency=currency, locale=locale,
                              cabinclass=cabinclass,
                              adults=adults,
                              model=model, cuda_ind=cuda_ind,
                              debug=debug,
                              errors=errors,
                              simplify_compute=simplify_compute,
                              underlyer=underlyer,
                              mt_ind=mt_ind,
                              price_by_range=price_by_range,
                              return_flight=return_flight,
                              recompute_ind=recompute_ind,
                              correct_drift=correct_drift,
                              write_data_progress=write_data_progress)
        # pickle.dump(res, open('/home/brumen/work/mrds/ao/tmp/res_1.obj', 'wb'))
    else:
        res = res_supplied
        
    # all simulation times 
    T_l_dep_num = construct_st(option_start_date, option_end_date, date_today_dt,
                               simplify_compute=simplify_compute)
    if return_flight:
        T_l_ret_num = construct_st(option_ret_start_date, option_ret_end_date, date_today_dt,
                                   simplify_compute=simplify_compute)
        
    if not return_flight:  # one-way flight
        penalty_used = penalty
        F_v_dep, F_mat_dep, flights_v_dep, reorg_flights_v_dep, s_v_dep, d_v_dep, valid_ind = res
    else:
        penalty_used = 2 * penalty
        (F_v_dep, F_v_ret), (F_mat_dep, F_mat_ret), \
            (flights_v_dep, flights_v_ret), (reorg_flights_v_dep, reorg_flights_v_ret), \
            (s_v_dep, s_v_ret), (d_v_dep, d_v_ret), valid_ind = res
        
    # sequential option parameter setup
    if len(F_v_dep) == 0 or (not valid_ind):
        opt_val_final = "Invalid"
        compute_all = False
    else:
        compute_all = True
        if not return_flight:
            F_v_used = F_v_dep
            F_mat_used = F_mat_dep
            s_v_used = s_v_dep
            d_v_used = d_v_dep
            T_l_used = T_l_dep_num
        else:
            F_v_used = (F_v_dep, F_v_ret)
            F_mat_used = (F_mat_dep, F_mat_ret)
            s_v_used = (s_v_dep, s_v_ret)
            d_v_used = (d_v_dep, d_v_ret)
            T_l_used = (T_l_dep_num, T_l_ret_num)
        opt_val_final = compute_option_raw(F_v_used, s_v_used, T_l_used, F_mat_used,
                                           K, penalty_used, p_c,
                                           rho, nb_sim=nb_sim,
                                           d_v=d_v_used,
                                           model=model, cuda_ind=cuda_ind,
                                           underlyer=underlyer,
                                           gen_first=gen_first)  #  * np.int(adults)
        opt_val_final['avg'] *= np.int(adults)
        
    # TO BE FURTHER IMPLEMENTED ??
    price_range = dict()
    if price_by_range and compute_all:  # compute_all guarantees there is something to compute
        complete_set_options = 3  # how many options to compute (default = 3)
        for ri in range(complete_set_options):
            outbound_date_consid = compute_date_by_fraction(date_today_dt, outbound_date_start,
                                                            complete_set_options-ri, complete_set_options)
            T_l_dep_num = construct_st(date_today, outbound_date_consid, date_today_dt,
                                       simplify_compute=simplify_compute)
            if not return_flight:
                T_l_used = T_l_dep_num
                key_ind = ds.convert_str_dateslash(outbound_date_consid)
            else:
                inbound_date_consid = compute_date_by_fraction(date_today_dt, inbound_date_start,
                                                               complete_set_options-ri, complete_set_options)
                T_l_ret_num = construct_st(date_today, inbound_date_consid, date_today_dt,
                                           simplify_compute=simplify_compute)
                T_l_used = (T_l_dep_num, T_l_ret_num)
                key_ind = ds.convert_str_dateslash(outbound_date_consid) + ' - ' + ds.convert_str_dateslash(inbound_date_consid)

            # for debugging 
            opt_val_scenario = compute_option_raw(F_v_used, s_v_used, T_l_used, F_mat_used,
                                                  K, penalty_used, p_c,
                                                  rho, nb_sim=nb_sim,
                                                  d_v=d_v_used,
                                                  model=model, cuda_ind=cuda_ind,
                                                  underlyer=underlyer)
            opt_val_scenario['avg'] *= np.int(adults)
            price_range[key_ind] = int(np.ceil(opt_val_scenario['avg']))

    if compute_all:
        if price_by_range:  # compute ranges
            if not return_flight:
                return opt_val_final, price_range, flights_v_dep, reorg_flights_v_dep, \
                    find_minmax_flight_subset(reorg_flights_v_dep, ret_ind=False)  # reorg_flights_v_dep
            else:
                return opt_val_final, price_range, (flights_v_dep, flights_v_ret), (reorg_flights_v_dep, reorg_flights_v_ret), \
                    find_minmax_flight_subset((reorg_flights_v_dep, reorg_flights_v_ret), ret_ind=True)
        else:  # dont compute range 
            if not return_flight:
                return opt_val_final, [], flights_v_dep, reorg_flights_v_dep, \
                    find_minmax_flight_subset(reorg_flights_v_dep, ret_ind=False)  # reorg_flights_v_dep
            else:
                return opt_val_final, [], (flights_v_dep, flights_v_ret), (reorg_flights_v_dep, reorg_flights_v_ret), \
                    find_minmax_flight_subset((reorg_flights_v_dep, reorg_flights_v_ret), ret_ind=True)
    else:
        return opt_val_final, [], [], [], []
