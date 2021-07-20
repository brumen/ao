""" Estimate the volatility and drift of air option stochastic processes
"""

import logging
import datetime
import numpy as np

from typing import List, Union

from ao.flight     import Flight, create_session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# summer months indicator (10 is not in).
SEASON_SUMMER = range(5, 10)


def date_in_season(date : Union[datetime.date, List[datetime.date]], season : str) -> Union[bool, List[bool]]:
    """ Checks whether date is in season

    :param date: date (or list of dates) to be checked whether they are in season.
    :param season: either 'summer' or 'winter'
    :returns: indicator if date is in the given season.
    """

    summer_cond = lambda d_: (d_.month in SEASON_SUMMER) if (season == 'summer') else (d_.month not in SEASON_SUMMER)

    if isinstance(date, datetime.date) or isinstance(date, datetime.datetime):  # date not list
        return summer_cond(date)

    # date is a list type object
    return [ summer_cond(d) for d in date]


def hour_in_section_atomic(hour : int, day_part : str) -> bool:
    """ Checks if the hour of departure is in a specific part of the day.

    :param hour: hour of departure, like 2, or something.
    :param day_part: part of the day, either 'morning', 'afternoon', 'evening', 'night'
    :returns: indicator whether the hour is in a specific part of the day.
    """

    if day_part == 'morning':
        return 6 <= hour < 11

    if day_part == 'afternoon':
        return 11 <= hour < 18

    if day_part == 'evening':
        return 18 <= hour < 23

    return 23 <= hour <= 24 and hour < 6


def hour_in_section(hour : Union[datetime.time, List[datetime.time]], section : str) -> Union[bool, List[bool]]:
    """ Checks whether the hour is in specified section of the day

    :param hour:    hour in datetime format or a list format
    :param section: either 'morning', 'afternoon', 'evening', 'night'
    """

    if isinstance(hour, datetime.time):
        return hour_in_section_atomic(hour.hour, section)

    return [hour_in_section_atomic(h.hour, section) for h in hour]


# TODO: CHECK THIS STUFF HERE!!
def flight_corr( origins   : List[str]
               , dests     : List[str]
               , carriers  : List[str]
               , dep_dates : List[datetime.date]
               , dcf        = 365.25 ):
    """ Compute the correlation between flights in the list

    :param origins: origin list of IATA airports, ['EWR'...]
    :param dests: destination list of IATA airports ['SFO', ...]
    :param carriers: list of carrier airlines (IATA codes) ['UA', 'B6'...]
    :param dep_dates: list of departure dates to consider
    :param dcf: day_count factor.
    :returns:
    """

    session = create_session()

    flights = session.query(Flight).filter( Flight.orig.in_(origins)
                                          , Flight.dest.in_(dests)
                                          , Flight.carrier.int_(carriers)
                                          , Flight.dep_date.in_(dep_dates)).all()

    prices = np.array([flight.price for flight in flights ])
    as_ofs = np.array([flight.as_of for flight in flights ])

    # TODO: Compute correlation
    return 0.5
