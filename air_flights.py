#
# Functions for air flights manipulation.
#

import datetime
import numpy as np
import logging

import ds
import ao_codes
import air_search
import ao_params

from ao_codes import MAX_TICKET


logger = logging.getLogger(__name__)


def find_minmax_ow(rof):
    """ Computes the min/max ticket by different subsets of tickets (e.g. days,
        hours, etc. ). This function only does that for one-way flights;
        adds fields 'min_max' to reorg_flights_v

    :param rof: reorganized flights
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
                             , ret_ind = False ) -> dict:
    """
    Finds the minimum and maximum of flights in each subset of flights

    :param reorg_flights_v: dictionary structure of flights TODO: DESCRIBE THE STRUCTURE.
    :type reorg_flights_v: dict or tuple(dict, dict)
    :param ret_ind:         indicator of return flight, either True/False
    :returns:               min_max subset over flights
    """

    if not ret_ind:  # outbound flight only
        return find_minmax_ow(reorg_flights_v)

    else:  # return flight
        return find_minmax_ow(reorg_flights_v[0]), find_minmax_ow(reorg_flights_v[1])


def obtain_flights_mat( flights
                      , flights_include
                      , date_today_dt : datetime.date ):
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


def sort_all( F_v : np.array
            , F_mat
            , s_v
            , d_v
            , fl_v
            , reorg_flights_v ):
    """
    Sorts the flights according to the F_v,
    _Important_ assmption being that similar flights by values are most correlated.

    regorg_flights_v is useless TODO: FIX LATER.

    :param F_v: vector of flight prices
    """

    return zip(*sorted(zip(F_v, F_mat, s_v, d_v, fl_v)))


def obtain_flights( origin_place : str
                  , dest_place   : str
                  , carrier      : str
                  , in_out_date_range
                  , flights_include
                  , cabinclass         = 'Economy'
                  , adults             = 1
                  , insert_into_livedb = True
                  , io_ind             = 'out'
                  , correct_drift      = True):
    """
    Get the flights for outbound and/or inbound flight

    :param origin_place:  origin of flights, IATA code (like 'EWR')
    :param dest_place:    dest of flights, IATA code (like 'SFO')
    :param carrier:       IATA code of the carrier considered
    :param in_out_date_range:   input/output date range _minus (with - sign)
                          output of function construct_date_range(outbound_date_start, outbound_date_end)
    :type in_out_date_range:    list of datetime.date
    :param io_ind:        inbound/outbound indicator ('in', 'out')
    :param correct_drift: whether to correct the drift, as described in the documentation
    :param cabinclass:    cabin class, one of 'Economy', ...
    """

    F_v, flights_v, F_mat, s_v_obtain, d_v_obtain = [], [], [], [], []

    reorg_flights_v = dict()

    # outbound, else inbound, reverse the origin, destination
    origin_used, dest_used = (origin_place, dest_place) if io_ind == 'out' else (dest_place, origin_place)

    for out_date in in_out_date_range:

        out_date_str = out_date.isoformat()

        yield out_date  # TODO: check this HERE
        ticket_val, flights, reorg_flights = \
            air_search.get_ticket_prices( origin_place       = origin_used
                                        , dest_place         = dest_used
                                        , outbound_date      = out_date
                                        , include_carriers   = carrier
                                        , cabinclass         = cabinclass
                                        , adults             = adults
                                        , insert_into_livedb = insert_into_livedb)

        # does the flight exist for that date??
        if out_date_str in reorg_flights:  # reorg_flights has string keys

            F_v.extend(ticket_val)
            io_dr_drift_vol = ao_params.get_drift_vol_from_db_precise( [x[1] for x in flights]  # just the departure time
                                                                     , origin_used
                                                                     , dest_used
                                                                     , carrier
                                                                     , correct_drift = correct_drift
                                                                     , fwd_value     = np.mean(ticket_val))

            s_v_obtain.extend([x[0] for x in io_dr_drift_vol])  # adding the vols
            d_v_obtain.extend([x[1] for x in io_dr_drift_vol])  # adding the drifts
            flights_v.extend(flights)
            F_mat.extend(obtain_flights_mat(flights, flights_include, datetime.date.today() ))  # maturity of forwards

            reorg_flights_v[out_date_str] = reorg_flights[out_date_str]

    if len(F_v) > 0:  # there are actual flights
        yield F_v, F_mat, s_v_obtain, d_v_obtain, flights_v, reorg_flights_v

    yield None  # no flights


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
                            , correct_drift = True ):
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
    """

    origin_used, dest_used = (origin_place, dest_place) if io_ind == 'out' else (dest_place, origin_place)

    F_v, flights_v, F_mat, s_v_obtain, d_v_obtain = [], [], [], [], []
    reorg_flights_v = dict()
    for outbound_date in io_dr_minus:  # io_dr_minus ... date range in datetime.date format
        # fliter prices from flights_include
        ticket_val = []
        flights = []  # (id, dep, arr, price, flight_id)
        reorg_flight = {}
        od_iso = outbound_date.isoformat()
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
                                       , datetime.date.today() ))  # maturity of forwards
        reorg_flights_v[od_iso] = reorg_flight

    return F_v, F_mat, s_v_obtain, d_v_obtain, flights_v, reorg_flights_v


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

    obtain_flights_f = obtain_flights_recompute if recompute_ind else obtain_flights

    if not return_flight:  # departure flights, always establish

        obtained_flights = obtain_flights_f( origin_place
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

        if not obtained_flights:  # if None, flights are not obtained.
            return None

        F_v_dep, F_mat_dep, s_v_dep, d_v_dep, flights_v_dep = sort_all(*obtained_flights)

    else:  # return flights handling

        # we have a restriction on which flights to include
        flights_include_dep, flights_include_ret = flights_include if flights_include else (None, None)

        obtained_flights_ret2 = obtain_flights_f( origin_place
                                                , dest_place
                                                , carrier
                                                , outbound_date_range
                                                , flights_include_dep
                                                , io_ind        = 'out'
                                                , correct_drift = correct_drift )

        if not obtained_flights_ret2:  # not valid, return immediately
            return None

        F_v_dep, F_mat_dep, s_v_dep_raw, d_v_dep_raw, flights_v_dep = sort_all(*obtained_flights_ret2)

        obtained_flights_ret = obtain_flights_f( origin_place
                                               , dest_place
                                               , carrier
                                               , inbound_date_range
                                               , flights_include_ret
                                               , io_ind        = 'in'
                                               , correct_drift = correct_drift )

        if not obtained_flights_ret:  # not valid, return immediately
            return None

        F_v_ret, F_mat_ret, s_v_ret_raw, d_v_ret_raw, flights_v_ret = sort_all(*obtained_flights_ret)

        valid_check = obtained_flights and obtained_flights_ret

        if valid_check:
            s_v_dep = s_v_dep_raw
            d_v_dep = d_v_dep_raw
            s_v_ret = s_v_ret_raw
            d_v_ret = d_v_ret_raw

    # TODO: reorg_flights_v_dep is in obtained_flights
    if valid_check:
        if not return_flight:
            return F_v_dep, F_mat_dep, flights_v_dep, reorg_flights_v_dep, s_v_dep, d_v_dep
        else:
            return (F_v_dep, F_v_ret),\
                   (F_mat_dep, F_mat_ret),\
                   (flights_v_dep, flights_v_ret),\
                   (reorg_flights_v_dep, reorg_flights_v_ret),\
                   (s_v_dep, s_v_ret),\
                   (d_v_dep, d_v_ret)

    return None  # not valid
