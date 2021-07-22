""" Functions handling the flights transformation from the server for recompute.
"""

import logging

from typing import Union, Dict, Tuple

import ao.ao_codes   as ao_codes

from ao.ao_codes   import MAX_TICKET


logger = logging.getLogger(__name__)


def find_minmax_ow(rof):
    """ Computes the min/max ticket by different subsets of tickets (e.g. days,
        hours, etc. ). This function only does that for one-way flights;
        adds fields 'min_max' to reorg_flights_v

    :param rof: reorganized flights
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


def find_minmax_flight_subset( reorg_flights_v : Union[Dict, Tuple[Dict, Dict]]) -> Union[Dict, Tuple[Dict, Dict]]:
    """ Finds the minimum and maximum of flights in each subset of flights

    :param reorg_flights_v: dictionary structure of flights TODO: DESCRIBE THE STRUCTURE.
    :returns: min_max subset over flights
    """

    ret_ind = True if isinstance(reorg_flights_v, tuple) else False  # indicator of return flight

    if not ret_ind:  # outbound flight only
        return find_minmax_ow(reorg_flights_v)

    return find_minmax_ow(reorg_flights_v[0]), find_minmax_ow(reorg_flights_v[1])


def filter_prices_and_flights( price_l
                             , flights_l
                             , reorg_flights_l
                             , flights_include) -> Union[None, Tuple]:
    """ fliter prices from flights_include

    :param price_l:
    :param flights_l:
    :param reorg_flights_l:
    :param flights_include: list of flights to include TODO: WHERE???
    :returns:
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

            return None  # invalid results.

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

    return F_v, flight_v, reorg_flight_v
