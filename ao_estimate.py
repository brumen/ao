# estimate the volatility and drift of air option stochastic processes
#

import logging
import datetime
import time

import numpy  as np
import pandas as pd

from typing      import List, Tuple

from ao.ao_db               import run_db_mysql
from ao.ao_codes            import LARGE_DRIFT
from ao.mysql_connector_env import MysqlConnectorEnv

from ao.flight import Flight, create_session


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmptyFlights(Exception):
    """ Exception for empty flights identification.
    """

    pass


def date_in_season(date : [datetime.date, List[datetime.date]], season : str):
    """
    checks whether date is in season
    :param date: given in date format, datetime format or any list related format 
    :param season: either 'summer' or 'winter'
    """

    cnd_outer = isinstance(date, datetime.date) or isinstance(date, datetime.datetime)

    if season == 'summer':
        if cnd_outer:
            return 5 <= date.month < 10

        # it's a list type object
        return [ 5 <= d.month < 10 for d in date]

    else:  # season == winter
        if cnd_outer: 
            return date.month <= 4 or date.month >= 10

        return [d.month <= 4 or d.month >= 10 for d in date]


def hour_in_section_atomic(hour_used, sect):
    """
    If ?? TODO: FINISH HERE

    """

    if sect == 'morning':
        return 6 <= hour_used < 11

    if sect == 'afternoon':
        return 11 <= hour_used < 18

    if sect == 'evening':
        return 18 <= hour_used < 23

    return 23 <= hour_used <= 24 and hour_used < 6


def hour_in_section(hour, section):
    """
    checks whether the hour is in specified section

    :param hour:    hour in datetime format or a list format
    :type hour:     dt.time or list[dt.time]
    :param section: either 'morning', 'afternoon', 'evening', 'night'
    :type section:  str
    """

    if type(hour) is datetime.time:
        return hour_in_section_atomic(hour.hour, section)

    # a list, go over
    return [hour_in_section_atomic(h.hour, section) for h in hour]


def flight_vol( orig    : str
              , dest    : str
              , carrier : str
              , as_of_date = None
              , insert_into_db    = False
              , model             = 'n'
              , correct_drift_vol = False
              , db_host = 'localhost') -> dict:
    """ Gets the flight for a particular origin, destination, carrier.

    :param orig: IATA code of the origin airport (e.g. SFO)
    :param dest: IATA code of the dest. airport (e.g. EWR)
    :param carrier: IATA code of the carrier airline (e.g. UA)
    :param as_of_date: the date of this computation.
    :param insert_into_db: indicator whether to insert the computed volatility in the database.
    :param model: normal ('n') or log-normal ('ln') model to compute drift/vol over
    :param db_host: mysql database host.
    :returns: dictionary where keys are tuples of day_hour, da TODO: FINISH HERE
    """

    result_dict = {}
    for dep_hour in ['morning', 'afternoon', 'evening', 'night']:
        for dep_day in ['weekday', 'weekend']:
            for dep_season in range(1,13):
                # print(dep_hour, dep_season, dep_day)  # morning, 2, weekend
                with MysqlConnectorEnv(host=db_host) as conn_mysql:
                    selected_flights = pd.read_sql_query("""SELECT DISTINCT flord.as_of, flord.price
                                                FROM flights_ord flord, flight_ids fids, reg_ids rid
                                                WHERE fids.orig= '{0}' AND fids.dest = '{1}' AND carrier='{2}' AND rid.tod='{3}' AND rid.month = {4} AND rid.weekday_ind = '{5}'
                                                AND flord.flight_id = fids.flight_id 
                                                AND rid.reg_id = flord.reg_id
                                                ORDER BY as_of""".format(orig, dest, carrier, dep_hour, dep_season, dep_day)
                                            , conn_mysql
                                            , parse_dates={'as_of': '%Y-%m-%d'})
                    # print (selected_flights)

                result_dict[(dep_hour, dep_day, dep_season)] = flight_vol_intern( selected_flights
                                                                                , model             = model
                                                                                , correct_drift_vol = correct_drift_vol )
                # TODO: TO FIX THIS!!
                if insert_into_db:
                    insert_into_db_function(as_of_date if as_of_date else time.localtime()
                                           , result_dict[(dep_hour, dep_day, dep_season)] )

    return result_dict


def flight_vol_intern( flights : pd.DataFrame
                     , model             = 'n'
                     , correct_drift_vol = False) -> [None, Tuple[float, float, int, float]]:
    """ Compute the volatility of particular flight between orig, dest and carrier

    :param flights: selected flights over which vol & drift is computed.
    :param model: indicator whether drift & vol should be computed for normal or log-normal model. ('n' or 'ln')
    :param correct_drift_vol: correct drift and vol: drift to 500. if negative,  computes the flight volatility and drift.
    :returns: None or tuple of (drift, volatility, length of volatility series, average price)
    """

    sel_flights = flights.drop_duplicates(subset='as_of')  # remove duplicates

    if len(sel_flights) <= 1:
        return None

    drift_len, drift, vol_1, vol_2, sum_price = compute_partial_drift_vol( sel_flights['as_of']
                                                                         , sel_flights['price']
                                                                         , model = model )

    drift    /= drift_len
    vol       = np.sqrt((vol_1/drift_len - (vol_2/drift_len)**2))
    avg_price = np.double(sum_price) / drift_len

    # correct drift if negative or has no drift results
    if correct_drift_vol and (drift <= 0):
        drift = LARGE_DRIFT  # large value

    return drift, vol, drift_len, avg_price


def compute_partial_drift_vol( date_l  : List[datetime.date]
                             , price_l : List[float]
                             , model = 'n'
                             , dcf = 385.25) -> Tuple[int, float, float, float, float]:
    """
    Compute the drift and volatility of the normal/lognormal model.

    :param date_l:  list of dates
    :param price_l: list of prices at those dates
    :param model:   model selected: 'n' for normal, 'ln' for log-normal
    :type model:    str
    :returns:       tuple of number of prices, drift computed,
                       vol computed E(X**2), vol_computed part E(X)
                       sum of prices
    """

    date_diff = np.diff(np.array([(x - datetime.datetime.now()).seconds for x in date_l])) / (dcf*86400)

    price_l_diff = np.diff(np.array(price_l))
    price_diff   = price_l_diff/np.array(price_l[:-1]) if model == 'ln' else price_l_diff

    drift_over_sqdate = price_diff/np.sqrt(date_diff)

    return ( len(price_l)
           , np.sum(price_diff/date_diff)
           , np.sum(drift_over_sqdate**2)
           , np.sum(drift_over_sqdate)
           , np.sum(price_l)
           , )


def all_vols_by_airline( carrier : str
                       , insert_into_db    = False
                       , as_of_date        = None
                       , model             = 'n'
                       , correct_drift_vol = False) -> dict:
    """ Estimates all drift/vol pairs for a particular airline.

    :param carrier: airline, carrier ('UA')
    :param insert_into_db: indicator whether to insert into db.
    :param as_of_date: the date we are running the estimation, if None, today's date/time.
    :param model: log-normal ('ln') or normal('n') model to estimate.
    :param correct_drift_vol: indicator whether to correct drift/volatility w/ large/small values.
    """

    flight_session = create_session()
    logger.info(f'Computing vols for carrier {carrier}')

    result_dict = {}
    for flight in flight_session.query(Flight).filter(Flight.carrier == carrier).all():
        try:
            orig = flight.orig
            dest = flight.dest
            result_dict[(orig, dest)] = flight_vol( orig
                                                  , dest
                                                  , carrier
                                                  , insert_into_db    = insert_into_db
                                                  , as_of_date        = as_of_date
                                                  , model             = model
                                                  , correct_drift_vol = correct_drift_vol)

        except EmptyFlights:  # if fails return empty
            logger.info(f'No flights between {orig} and {dest} for carrier {carrier}')

        except Exception as e:
            raise e

    return result_dict


def all_vols( insert_into_db    = False
            , as_of_date        = None
            , model             = 'n'
            , correct_drift_vol = False ) -> dict:
    """ Estimates all drift/vol pairs for all airlines.

    :param insert_into_db: indicator whether to insert into db.
    :param as_of_date: the date we are running the estimation, if None, today's date/time.
    :param model: log-normal ('ln') or normal('n') model to estimate.
    :param correct_drift_vol: indicator whether to correct drift/volatility w/ large/small values.
    """

    return {carrier: all_vols_by_airline( carrier           = str(carrier)
                                        , insert_into_db    = insert_into_db
                                        , as_of_date        = as_of_date
                                        , model             = model
                                        , correct_drift_vol = correct_drift_vol )
            for carrier, _ in run_db_mysql("SELECT DISTINCT iata_code FROM iata_codes") }


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