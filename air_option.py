# air option computation file
import config
import datetime as dt
import numpy as np
import time
import json
import logging

if config.CUDA_PRESENT:
    import cuda_ops
    import pycuda.gpuarray as gpa

import vols.vols as vols
import mc
import ds
import ao_codes
import air_search
import ao_params

from ao_codes import MAX_TICKET, MIN_PRICE


logger = logging.getLogger(__name__)  # root logger


def date_today():
    """
    returns today's date
    :returns:  today's date
    :rtype:    datetime.date
    """

    lt = time.localtime()
    return ds.convert_str_date(str(lt.tm_year) +
                               str(ds.d2s(lt.tm_mon)) +
                               str(ds.d2s(lt.tm_mday)))


def data_yield(data_dict):
    """
    Returns the data_dict in the form that server understands

    :param data_dict: dictionary to be sent to the browser
    :type data_dict: dict
    :returns: string that the server understands
    :rtype: str
    """

    # return "data: {0}\n\n".format(json.dumps(data_dict))
    return json.dumps(data_dict)


def air_option( F_v
              , s_v
              , d_v
              , T_l
              , T_mat
              , K
              , nb_sim    = 1000
              , rho       = 0.9
              , cuda_ind  = False
              , underlyer ='n' ):
    """
    Computes the value of the air option with low memory impact.

    :param F_v: vector of forward prices, or a tuple (F_1_v, F_2_v) for return flights
    :type F_v: np.array or (np.array, np.array)
    :param s_v: vector of vols, or a tuple for return flights, similarly to F_v
    :type s_v: np.array or (np.array, np.array)
    :param T_l: simulation list, same as s_v
    :type T_l: np.array or (np.array, np.array)
    :param T_mat: maturity list 
    :param K: strike price
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

    args = [ F_v, s_v, d_v, T_l, rho_m, nb_sim, T_mat ]
    kwargs = { 'model'   : underlyer
             , 'cuda_ind': cuda_ind}

    mc_used = mc.mc_mult_steps if not return_flight_ind else mc.mc_mult_steps_ret

    F_max = mc_used(*args, **kwargs)  # simulation of all flight prices

    # final option payoff
    if not cuda_ind:
        return np.mean(np.maximum (np.amax(F_max, axis=0) - K, 0.))
    else:
        return np.mean(gpa.maximum(cuda_ops.amax_gpu_0(F_max) - K, 0.))


def compute_option_raw( F_v
                      , s_v
                      , d_v
                      , T_l_num
                      , T_mat_num
                      , K
                      , rho
                      , nb_sim    = 10000
                      , cuda_ind  = False
                      , underlyer = 'n' ):
    """
    Computes the value of the option sequentially, in order to minimize memory footprint

    :param F_v:       vector of tickets for one-way flights, tuple for return flights
    :type F_v:        np.array or tuple(np.array, np.array)
    :param s_v:       vector of vols for one-way, the same size as F_v, or tuple for return flights
    :type s_v:        np.array or tuple(np.array, np.array)
    :param d_v:       vector of drifts for one-way, the same size as F_v, or tuple for return flights
    :type d_v:        np.array or tuple(np.array, np.array)
    :param T_l_num:   simulation times of for tickets; or a tuple for departure, return tickets
    :type T_l_num:    np.array 1-dimensional; or tuple (np.array, np.array)
    :param T_mat_num: maturity of tickets TODO
    :type T_mat_num:
    :param K:         strike of the option
    :type K:          double
    :param rho:       correlation matrix for flight tickets, or a tuple of matrices for departure, return tickets
    :type rho:        np.array 2 dimensional; or a tuple of two such matrices
    :param nb_sim:    number of simulations
    :type nb_sim:     integer
    :param cuda_ind:  whether to use cuda; True or False
    :type cuda_ind:   bool
    :param underlyer: which model to use - lognormal or normal ('ln' or 'n')
    :type underlyer:  string; 'ln' or 'n'
    :returns:
    :rtype:
    """

    opt_val_final = air_option( F_v
                              , s_v
                              , d_v
                              , T_l_num
                              , T_mat_num
                              , K
                              , nb_sim    = nb_sim
                              , rho       = rho
                              , cuda_ind  = cuda_ind
                              , underlyer = underlyer )

    # markups to the option value
    percentage_markup = ao_codes.reserves + ao_codes.tax_rate # 10 % markup 

    if type(F_v) is not tuple:
        F_v_max = np.max(F_v)
    else:
        F_v_max = max(np.max(F_v[0]), np.max(F_v[1]))

    # minimal payoff
    min_payoff = max(MIN_PRICE, F_v_max / ao_codes.ref_base_F * MIN_PRICE)

    return max(min_payoff, (1. + percentage_markup) * opt_val_final)
    

def construct_sim_times( date_start
                       , date_end
                       , date_today_dt
                       , simplify_compute
                       , dcf = 365. ):
    """
    Constructs the simulation times used in Monte Carlo

    :param date_start:    date start
    :type date_start:     datetime.date
    :param date_end:      date end
    :type date_end:       datetime.date
    :param date_today_dt: date today
    :type date_today_dt:  datetime.date
    :param dcf:           day count factor
    :type dcf:            double
    :returns:             list of simulated dates
    :rtype:               list of datetime.date
    """
    T_l = ds.construct_date_range(date_start, date_end)  # in date format

    if simplify_compute == 'all_sim_dates':
        return [(date_sim - date_today_dt).days/dcf for date_sim in T_l]
    elif simplify_compute == 'take_last_only':
        return [(T_l[-1] - date_today_dt).days/dcf]


def find_minmax_ow(rof):
    """
    computes the min/max ticket by different subsets of tickets (e.g. days,
        hours, etc. ). This function only does that for one-way flights;
        adds fields 'min_max' to reorg_flights_v

    :param rof: TODO HERE
    :type rof:  TODO HERE
    """
    min_max_dict = dict()  # new dict to return
    change_dates = rof.keys()
    total_min, total_max = MAX_TICKET, 0.

    for c_date in change_dates:
        min_max_dict[c_date] = dict()
        flights_by_daytime = rof[c_date]
        flight_daytimes = rof[c_date].keys()
        cd_min, cd_max = MAX_TICKET, 0.

        for f_daytime in flight_daytimes:
            flight_subset = flights_by_daytime[f_daytime]
            # now find minimum or maximum
            min_subset, max_subset = MAX_TICKET, 0.

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


def find_minmax_flight_subset( reorg_flights_v
                             , ret_ind = False):
    """
    Finds the minimum and maximum of flights in each subset of flights

    :param reorg_flights_v: dictionary structure of flights
    :type reorg_flights_v:  dict
    :param ret_ind:         indicator of return flight
    :type ret_ind:          bool
    :returns:               min_max subset over flights
    :rtype:                 dict
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
    """
    drift structure of the model,

    :param d: drift at time t
    :type d:  double
    :param t: time at which drift is evaluted
    :type t:  double
    :returns: drift of the model
    :rtype:   double
    """

    return d


def obtain_flights_mat( flights
                      , flights_include
                      , date_today_dt):
    """
    Constucting the flight maturity, with censoring the flights that are not included in the flights_include list

    :param flights: list of flights to be included in the construction of flights maturity
    :type flights: list of [(id, dep, arr, price, flight_nb)...]
    :param flights_include: flights to be included in the TODO:
    :type flights_include: dict of flights as in reorg_flights,
    :param date_today_dt: today's date in datetime format
    :type date_today_dt: datetime.date
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
    Sorts the flights according to the F_v,
    assmption being that similar flights by values are most correlated

    :param F_v: vector of flight prices
    :type F_v:  np.array

    """

    zip_ls = sorted(zip(F_v, F_mat, s_v, d_v, fl_v))
    F_v_s, F_mat_s, s_v_s, d_v_s, fl_v_s = zip(*zip_ls)

    return F_v_s, F_mat_s, s_v_s, d_v_s, fl_v_s


def obtain_flights( origin_place
                  , dest_place
                  , carrier
                  , in_out_date_range
                  , flights_include
                  , cabinclass         = 'Economy'
                  , adults             = 1
                  , insert_into_livedb = True
                  , io_ind             = 'out'
                  , correct_drift      = True
                  , publisher_ao       = None ):
    """
    Get the flights for outbound and/or inbound flight

    :param origin_place:  origin of flights, IATA code (like 'EWR')
    :type origin_place:   str
    :param dest_place:    dest of flights, IATA code (like 'SFO')
    :type dest_place:     str
    :param carrier:       IATA code of the carrier considered
    :type carrier:        str
    :param in_out_date_range:   input/output date range _minus (with - sign)
                          output of function construct_date_range(outbound_date_start, outbound_date_end)
    :type in_out_date_range:    list of datetime.date
    :param io_ind:        inbound/outbound indicator ('in', 'out')
    :type io_ind:         str
    :param cabinclass:    cabin class, one of 'Economy', ...
    :type cabinclass:     str
    :param correct_drift: whether to correct the drift, as described in the documentation
    :type correct_drift:  bool
    """

    F_v, flights_v, F_mat, s_v_obtain, d_v_obtain = [], [], [], [], []
    reorg_flights_v = dict()

    if io_ind == 'out':  # outbound
        origin_used, dest_used = origin_place, dest_place
    else:  # inbound, reverse the origin, destination
        origin_used, dest_used = dest_place, origin_place

    for out_date in in_out_date_range:

        out_date_str = out_date.isoformat()
        logger.info(';'.join([ 'AO'
                              ,  json.dumps( {'finished': False,
                                              'results' : 'Fetching flights for ' + out_date_str} ) ]) )

        if publisher_ao:
            publisher_ao.publish(data_yield({ 'finished': False
                                            , 'result'  : 'Fetching flights for ' + out_date_str}))

        ticket_val, flights, reorg_flights = \
            air_search.get_ticket_prices( origin_place       = origin_used
                                        , dest_place         = dest_used
                                        , outbound_date      = out_date
                                        , include_carriers   = carrier
                                        , cabinclass         = cabinclass
                                        , adults             = adults
                                        , insert_into_livedb = insert_into_livedb)

        logger.info(';'.join([ 'AO'
                              , json.dumps({'finished'    : False,  # is_return_for_writing and last_elt,
                                            'results': ' '.join(["Fetched flights for", out_date_str]) }) ] ) )

        if publisher_ao:
            publisher_ao.publish(data_yield({ 'finished': False
                                            , 'result'  : ' '.join(["Fetched flights for", out_date_str ] ) } ) )

        # does the flight exist for that date??
        if out_date_str in reorg_flights:  # reorg_flights has string keys

            F_v.extend(ticket_val)
            io_dr_drift_vol = ao_params.get_drift_vol_from_db_precise( map(lambda x: x[1], flights) # just the departure time
                                                                     , origin_used
                                                                     , dest_used
                                                                     , carrier
                                                                     , correct_drift = correct_drift
                                                                     , fwd_value     = np.mean(ticket_val))

            s_v_obtain.extend([x[0] for x in io_dr_drift_vol])  # adding the vols
            d_v_obtain.extend([x[1] for x in io_dr_drift_vol])  # adding the drifts
            flights_v.extend(flights)
            F_mat.extend(obtain_flights_mat(flights, flights_include, date_today()))  # maturity of forwards
            reorg_flights_v[out_date_str] = reorg_flights[out_date_str]

    F_v = np.array(F_v)
    F_mat = np.array(F_mat)

    if len(F_v) > 0:  # there are actual flights
        return F_v, F_mat, s_v_obtain, d_v_obtain, flights_v, reorg_flights_v, 'Valid'
    else:  # no flights, indicate that it is wrong
        return [], [], [], [], [], [], 'Invalid'


def filter_prices_and_flights( price_l
                             , flights_l
                             , reorg_flights_l
                             , flights_include):
    """
    fliter prices from flights_include

    :param price_l:
    :type price_l:
    :param flights_l:
    :type flights_l:
    :param reorg_flights_l:
    :type reorg_flights_l:
    :param flights_include: list of flights to include TODO: WHERE???
    :type flights_include:
    :returns:
    :rtype:
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


def obtain_flights_recompute( origin_place
                            , dest_place
                            , carrier
                            , io_dr_minus
                            , flights_include
                            , cabinclass         = 'Economy'
                            , adults             = 1
                            , insert_into_livedb = True
                            , io_ind        = 'out'
                            , correct_drift = True
                            , publisher_ao  = None):
    """
    Get the flights for recompute method.
    IMPORTANT: cabinclass, adults, insert_into_livedb HAVE TO BE THERE, to correspond to obtain_flights fct.

    :param origin_place:
    :param io_dr_minus:     input/output date range _minus (with - sign)
    :type io_dr_minus:
    :param flights_include: flights to be considered for recomputation
    :type flights_include:  TODO
    :param io_ind:          inbound/outbound indicator ('in', or 'out')
    :type io_ind:           str
    :param publisher_ao:    publisher object, _NOT_ used by this function, _ONLY_ HERE to be
                                compatible w/ obtain_flights
    :type publisher_ao:     sse.Publisher
    """

    F_v, flights_v, F_mat, s_v_obtain, d_v_obtain = [], [], [], [], []
    reorg_flights_v = dict()
    if io_ind == 'out':  # outbound
        origin_used, dest_used = origin_place, dest_place
    else:
        origin_used, dest_used = dest_place, origin_place

    for od in io_dr_minus:  # od ... outbound date, io_dr_minus ... date range in datetime.date format
        # fliter prices from flights_include
        ticket_val = []
        flights = []  # (id, dep, arr, price, flight_id)
        reorg_flight = {}
        od_iso = od.isoformat()
        for tod in flights_include[od_iso]:  # iterating over time of day
            reorg_flight[tod] = {}
            for dep_time in flights_include[od_iso][tod]:
                res = flights_include[od_iso][tod][dep_time]
                if dep_time != 'min_max':
                    flight_id, _, dep_time, arr_date, arr_time, flight_price, flight_id, flight_included = res
                    carrier_tmp = flight_id[:2]  # first two letters of id - somewhat redundant
                    if flight_included:
                        ticket_val.append(flight_price)
                        flights.append((flight_id,
                                        od_iso + 'T' + dep_time,
                                        arr_date + 'T' + arr_time,
                                        flight_price,
                                        flight_id))
                        reorg_flight[tod][dep_time] = flights_include[od_iso][tod][dep_time]

        # add together
        F_v.extend(ticket_val)
        flight_dep_time_added = [x[1] for x in flights]  # just the departure time
        io_dr_drift_vol = ao_params.get_drift_vol_from_db_precise( flight_dep_time_added
                                                                 , origin_used
                                                                 , dest_used
                                                                 , carrier
                                                                 , correct_drift = correct_drift
                                                                 , fwd_value     = np.mean(ticket_val))

        s_v_obtain.extend([x[0] for x in io_dr_drift_vol])  # adding the vols
        d_v_obtain.extend([x[1] for x in io_dr_drift_vol])  # adding the drifts
        flights_v.extend(flights)
        F_mat.extend(obtain_flights_mat( flights
                                       , flights_include
                                       , date_today()))  # maturity of forwards
        reorg_flights_v[od_iso] = reorg_flight

    return np.array(F_v), np.array(F_mat), s_v_obtain, d_v_obtain, flights_v, reorg_flights_v, 'Valid'


def get_flight_data( flights_include     = None
                   , origin_place        = 'SFO'
                   , dest_place          = 'EWR'
                   , outbound_date_start = None
                   , outbound_date_end   = None
                   , inbound_date_start  = None
                   , inbound_date_end    = None
                   , carrier             = 'UA'
                   , cabinclass          = 'Economy'
                   , adults              = 1
                   , return_flight       = False
                   , recompute_ind       = False
                   , correct_drift       = True
                   , insert_into_livedb  = True
                   , publisher_ao        = None ):
    """
    Get flight data for the parameters specified

    :param flights_include:      if None - include all
                                 if specified, then only consider the flights in flights_include
    :type flights_include:       list of flights # TODO: SPECIFY THIS BETTER
    :param origin_place:         IATA code of the origin airport, e.g.  'SFO'
    :type origin_place:          string
    :param dest_place:           IATA code of the destination airport, e.g.  'SFO'
    :type dest_place:            string
    :param outbound_date_start:  start date of the outbound flights
    :type outbound_date_start:   datetime.date
    :param outbound_date_end:    end date of the outbound flights
    :type outbound_date_end:     datetime.date
    :param inbound_date_start:   start date of the inbound (return) flights
    :type inbound_date_start:    datetime.date
    :param inbound_date_end:     end date of the inbound (return) flights
    :type inbound_date_end:      datetime.date
    :param carrier:              IATA code of the carrier, e.g. 'UA'
    :type carrier:               string

    :param publisher_ao:         publisher object (from sse) for Air options,
    :type publisher_ao:          sse.Publisher

    """

    # outbound data range
    outbound_date_range = ds.construct_date_range(outbound_date_start, outbound_date_end)

    if return_flight:
        inbound_date_range = ds.construct_date_range(inbound_date_start, inbound_date_end)

    if recompute_ind:
        obtain_flights_f = obtain_flights_recompute
    else:
        obtain_flights_f = obtain_flights
        
    # departure flights, always establish
    if not return_flight:

        if publisher_ao:
            publisher_ao.publish(data_yield({ 'finished': False
                                            , 'result' : 'Fetching outbound data.'}))

        F_v_dep_uns, F_mat_dep_uns,\
            s_v_dep_u_uns, d_v_dep_u_uns,\
            flights_v_dep_uns, reorg_flights_v_dep,\
            valid_check = obtain_flights_f( origin_place
                                          , dest_place
                                          , carrier
                                          , outbound_date_range
                                          , flights_include
                                          , cabinclass            = cabinclass
                                          , adults                = adults
                                          , insert_into_livedb    = insert_into_livedb
                                          , io_ind                = 'out'
                                          , correct_drift         = correct_drift
                                          , publisher_ao          = publisher_ao )

        if publisher_ao:
            publisher_ao.publish(data_yield({ 'finished': False
                                            , 'result'  : 'Finished outbound flights fetch.' } ))

        if valid_check != 'Valid':  # not valid, return immediately
            if publisher_ao:
                publisher_ao.publish(data_yield({ 'finished': True
                                                , 'result'  : 'Outbound flight error.'}))
            return [], [], [], [], [], [], False
        
        F_v_dep, F_mat_dep, s_v_dep, d_v_dep, \
            flights_v_dep = sort_all( F_v_dep_uns
                                    , F_mat_dep_uns
                                    , s_v_dep_u_uns
                                    , d_v_dep_u_uns
                                    , flights_v_dep_uns)

        F_v_dep = np.array(F_v_dep)  # these are np.arrays, correct back
        F_mat_dep = np.array(F_mat_dep)

    else:  # return flights handling

        if flights_include:  # we have a restriction on which flights to include
            flights_include_dep, flights_include_ret = flights_include
        else:  # all flights taken for computation
            flights_include_dep, flights_include_ret = None, None

        F_v_dep_uns      , F_mat_dep_uns      , \
        s_v_dep_raw_uns  , d_v_dep_raw_uns    , \
        flights_v_dep_uns, reorg_flights_v_dep, \
        valid_check_out = obtain_flights_f( origin_place
                                          , dest_place
                                          , carrier
                                          , outbound_date_range
                                          , flights_include_dep
                                          , io_ind        = 'out'
                                          , correct_drift = correct_drift
                                          , publisher_ao  = publisher_ao)

        if valid_check_out != 'Valid':  # not valid, return immediately
            return ([], []), ([], []), ([], []), ([], []), ([], []), ([], []), False

        F_v_dep    , F_mat_dep  , \
        s_v_dep_raw, d_v_dep_raw, \
        flights_v_dep = sort_all( F_v_dep_uns
                                , F_mat_dep_uns
                                , s_v_dep_raw_uns
                                , d_v_dep_raw_uns
                                , flights_v_dep_uns)
        F_v_dep = np.array(F_v_dep)  # these are np.arrays, correct back
        F_mat_dep = np.array(F_mat_dep)

        F_v_ret_uns, F_mat_ret_uns, \
        s_v_ret_raw_uns, d_v_ret_raw_uns, \
        flights_v_ret_uns, reorg_flights_v_ret, \
        valid_check_in = obtain_flights_f(origin_place
                                          , dest_place
                                          , carrier
                                          , inbound_date_range
                                          , flights_include_ret
                                          , io_ind        = 'in'
                                          , correct_drift = correct_drift
                                          , publisher_ao  = publisher_ao)

        if valid_check_in != 'Valid':  # not valid, return immediately
            return ([], []), ([], []), ([], []), ([], []), ([], []), ([], []), False

        F_v_ret, F_mat_ret, \
        s_v_ret_raw, d_v_ret_raw, \
        flights_v_ret = sort_all(F_v_ret_uns
                                 , F_mat_ret_uns
                                 , s_v_ret_raw_uns
                                 , d_v_ret_raw_uns
                                 , flights_v_ret_uns)

        F_v_ret = np.array(F_v_ret)  # these are np.arrays, correct back
        F_mat_ret = np.array(F_mat_ret)

        valid_check = (valid_check_out == 'Valid') and (valid_check_in == 'Valid')

        if valid_check:
            s_v_dep = s_v_dep_raw
            d_v_dep = d_v_dep_raw
            s_v_ret = s_v_ret_raw
            d_v_ret = d_v_ret_raw

    if valid_check:
        if not return_flight:
            return F_v_dep, F_mat_dep, flights_v_dep, reorg_flights_v_dep, s_v_dep, d_v_dep, True
        else:
            return (F_v_dep, F_v_ret), (F_mat_dep, F_mat_ret), \
                (flights_v_dep, flights_v_ret), \
                (reorg_flights_v_dep, reorg_flights_v_ret), \
                (s_v_dep, s_v_ret), (d_v_dep, d_v_ret), True

    else:  # not valid

        if not return_flight:
            return [], [], [], [], [], [], False
        else:
            return ([], []), ([], []), ([], []), ([], []), ([], []), ([], []), False


def compute_date_by_fraction(dt_today, dt_final, fract, total_fraction):
    """
    Computes the date between dt_today and dt_final where the days between
    dt_today is the fract of dates between dt_today and dt_final


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
    # - 3 ... no change in the last 3 days
    return dt_today + dt.timedelta(days= (dt_final - dt_today).days * fract/total_fraction - 3)


def compute_option_val( origin_place          = 'SFO'
                      , dest_place            = 'EWR'
                      , flights_include       = None
                      # when can you change the option
                      , option_start_date     = None
                      , option_end_date       = None
                      , option_ret_start_date = None
                      , option_ret_end_date   = None
                      # next 4 - when do the (changed) flights occur
                      , outbound_date_start   = None
                      , outbound_date_end     = None
                      , inbound_date_start    = None
                      , inbound_date_end      = None
                      , K                     = 1600.
                      , carrier               = 'UA'
                      , nb_sim                = 10000
                      , rho                   = 0.95
                      , adults                = 1
                      , cabinclass            = 'Economy'
                      , cuda_ind              = False
                      , simplify_compute      = 'take_last_only'
                      , underlyer             = 'n'
                      , price_by_range        = True
                      , return_flight         = False
                      , flights_supplied      = None
                      , recompute_ind         = False
                      , correct_drift         = True
                      , publisher_ao          = False ):
    """
    computes the flight option

    :param origin_place:            IATA code of the origin airport ('SFO')
    :type origin_place:             str
    :param dest_place:              IATA code of the destination airport ('EWR')
    :type dest_place:               str
    :param flights_include:         list of flights to include in pricing this option
    :type flights_include:          list of tuples # TODO: BE MORE PRECISE HERE
    :param option_start_date:       the date when you can start changing the outbound flight
    :type option_start_date:        datetime.date
    :param option_end_date:         the date when you stop changing the outbound flight
    :type option_end_date:          datetime.date
    :param option_ret_start_date:   the date when you can start changing the inbound flight
    :type option_ret_start_date:    datetime.date
    :param option_ret_end_date:     the date when you stop changing the outbound flight
    :type option_ret_end_date:      datetime.date
    :param outbound_date_start:     start date for outbound flights to change to
    :type outbound_date_start:      datetime.date
    :param outbound_date_end:       end date for outbound flights to change to
    :type outbound_date_end:        datetime.date
    :param inbound_date_start:      start date for inbound flights to change to
    :type inbound_date_start:       datetime.date
    :param inbound_date_end:        end date for inbound flights to change to
    :type inbound_date_end:         datetime.date
    :param K:                       option strike
    :type K:                        double
    :param carrier:                 IATA code of the carrier
    :type carrier:                  str
    :param nb_sim:                  number of simulations
    :type nb_sim:                   int
    :param rho:                     correlation between flights parameter
    :type rho:                      double
    :param adults:                  nb. of people on this ticket
    :type adults:                   int
    :param cabinclass:              class of flight ticket
    :type cabinclass:               str
    :param cuda_ind:                whether to use cuda for computation
    :type cuda_ind:                 bool
    :param simplify_compute:        simplifies the computation in that it only simulates the last simulation date
    :type simplify_compute:         str, options are: "take_last_only", "all_sim_dates"
    :param publisher_ao:            publisher object where the function can publish its ongoing
    :type publisher_ao:             sse.Publisher
    """

    # date today 
    date_today_dt = date_today()

    if flights_supplied is None:  # no flights are supplied, find them

        if publisher_ao:
            publisher_ao.publish(data_yield({ 'finished': False
                                            , 'result'  : 'Initiating flight search.'  }))

        flights = get_flight_data( flights_include       = flights_include
                                 , origin_place          = origin_place
                                 , dest_place            = dest_place
                                 , outbound_date_start   = outbound_date_start
                                 , outbound_date_end     = outbound_date_end
                                 , inbound_date_start    = inbound_date_start
                                 , inbound_date_end      = inbound_date_end
                                 , carrier               = carrier
                                 , cabinclass            = cabinclass
                                 , adults                = adults
                                 , return_flight         = return_flight
                                 , recompute_ind         = recompute_ind
                                 , correct_drift         = correct_drift
                                 , publisher_ao          = publisher_ao )

    else:  # flights are provided, use these
        flights = flights_supplied
        
    # all simulation times 
    T_l_dep_num = construct_sim_times( option_start_date
                                     , option_end_date
                                     , date_today_dt
                                     , simplify_compute = simplify_compute)
    if return_flight:
        T_l_ret_num = construct_sim_times( option_ret_start_date
                                         , option_ret_end_date
                                         , date_today_dt
                                         , simplify_compute = simplify_compute)
        
    if not return_flight:  # one-way flight
        F_v_dep, F_mat_dep, flights_v_dep, reorg_flights_v_dep, s_v_dep, d_v_dep, valid_ind = flights

    else:  # return flight
        (F_v_dep, F_v_ret), (F_mat_dep, F_mat_ret), \
            (flights_v_dep, flights_v_ret), (reorg_flights_v_dep, reorg_flights_v_ret), \
            (s_v_dep, s_v_ret), (d_v_dep, d_v_ret), valid_ind = flights
        
    # sequential option parameter setup
    if len(F_v_dep) == 0 or (not valid_ind):  # len
        opt_val_final = "Invalid"
        compute_all = False

    else:
        compute_all = True
        if not return_flight:  # one way flight
            F_v_used = F_v_dep
            F_mat_used = F_mat_dep
            s_v_used = s_v_dep
            d_v_used = d_v_dep
            T_l_used = T_l_dep_num

        else:  # return flight
            F_v_used = (F_v_dep, F_v_ret)
            F_mat_used = (F_mat_dep, F_mat_ret)
            s_v_used = (s_v_dep, s_v_ret)
            d_v_used = (d_v_dep, d_v_ret)
            T_l_used = (T_l_dep_num, T_l_ret_num)

        opt_val_final = compute_option_raw( F_v_used
                                          , s_v_used
                                          , d_v_used
                                          , T_l_used
                                          , F_mat_used
                                          , K
                                          , rho
                                          , nb_sim     = nb_sim
                                          , cuda_ind   = cuda_ind
                                          , underlyer  = underlyer )\
                        * np.int(adults)

    # construct the price range
    price_range = dict()
    if price_by_range and compute_all:  # compute_all guarantees there is something to compute
        complete_set_options = 3  # how many options to compute (default = 3)
        for ri in range(complete_set_options):
            outbound_date_consid = compute_date_by_fraction( date_today_dt
                                                           , outbound_date_start
                                                           , complete_set_options-ri
                                                           , complete_set_options)
            T_l_dep_num = construct_sim_times( date_today_dt
                                             , outbound_date_consid
                                             , date_today_dt
                                             , simplify_compute = simplify_compute)

            if not return_flight:
                T_l_used = T_l_dep_num
                key_ind = ds.convert_datetime_str(outbound_date_consid)
            else:
                inbound_date_consid = compute_date_by_fraction( date_today_dt
                                                              , inbound_date_start
                                                              , complete_set_options-ri
                                                              , complete_set_options)
                T_l_ret_num = construct_sim_times( date_today_dt
                                                 , inbound_date_consid
                                                 , date_today_dt
                                                 , simplify_compute = simplify_compute)
                T_l_used = (T_l_dep_num, T_l_ret_num)
                key_ind = ds.convert_datetime_str(outbound_date_consid) + ' - ' + ds.convert_datetime_str(inbound_date_consid)

            # for debugging
            opt_val_scenario = compute_option_raw( F_v_used
                                                 , s_v_used
                                                 , d_v_used
                                                 , T_l_used
                                                 , F_mat_used
                                                 , K
                                                 , rho
                                                 , nb_sim    = nb_sim
                                                 , cuda_ind  = cuda_ind
                                                 , underlyer = underlyer)\
                              * np.int(adults)
            price_range[key_ind] = int(np.ceil(opt_val_scenario))

    if compute_all:

        if price_by_range:  # compute ranges
            if not return_flight:  # one-way
                return opt_val_final, price_range, flights_v_dep, reorg_flights_v_dep, \
                    find_minmax_flight_subset(reorg_flights_v_dep, ret_ind=False)
            else:  # return
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
