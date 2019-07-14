# estimate the volatility and drift of air option stochastic processes
#

import logging
import datetime
import time
import sqlite3

import numpy as np
import pandas as pd

from typing      import List, Tuple

from ao_db               import SQLITE_FILE, run_db_mysql
from ao_codes            import LARGE_DRIFT, DCF
from mysql_connector_env import MysqlConnectorEnv


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
              , db_host = 'localhost'):
    """ Gets the flight for a particular origin, destination, carrier.

    :param orig: IATA code of the origin airport (e.g. SFO)
    :param dest: IATA code of the dest. airport (e.g. EWR)
    :param carrier: IATA code of the carrier airline (e.g. UA)
    :param as_of_date: the date of this computation.
    :param insert_into_db: indicator whether to insert the computed volatility in the database.
    :param model: normal ('n') or log-normal ('ln') model to compute drift/vol over
    :param db_host: mysql database host.
    """

    for dep_hour in ['morning', 'afternoon', 'evening', 'night']:
        for dep_day in ['weekday', 'weekend']:
            for dep_season in range(1,13):
                with MysqlConnectorEnv(host=db_host) as conn_mysql:
                    selected_flights = pd.read_sql_query("""SELECT DISTINCT flord.as_of, fids.orig, fids.dest, flord.price, rid.tod, rid.month, rid.weekday_ind
                                                FROM flights_ord flord, flight_ids fids, reg_ids rid
                                                WHERE orig= '{0}' AND dest = '{1}' AND carrier='{2}' AND tod='{3}' AND month = '{4}' AND weekday_ind = '{5}'
                                                AND flord.flight_id = fids.flight_id 
                                                AND rid.reg_id = flord.reg_id
                                                ORDER BY as_of""".format(orig, dest, carrier, dep_hour, dep_season, dep_day)
                                            , conn_mysql
                                            , parse_dates={'as_of': '%Y-%m-%d'})

                print (flight_vol_intern( selected_flights
                                 , model             = model
                                 , correct_drift_vol = correct_drift_vol ))
                # TODO: TO FIX THIS!!
                if insert_into_db:
                    insert_into_db_function(as_of_date if as_of_date else time.localtime())


def flight_vol_intern( flights : pd.DataFrame
                     , model             = 'n'
                     , correct_drift_vol = False):
    """ Compute the volatility of particular flight between orig, dest and carrier

    :param flights: selected flights over which vol & drift is computed.
    :param model: indicator whether drift & vol should be computed for normal or log-normal model. ('n' or 'ln')
    :param correct_drift_vol: correct drift and vol: drift to 500. if negative,  computes the flight volatility and drift.
    """

    drift        = 0.  # this will be incrementally added
    drift_len    = 0
    vol_1, vol_2 = 0., 0.
    sum_price    = 0.

    sel_flights           = flights.drop_duplicates(subset='as_of')
    sel_flights_dep_dates = sel_flights['as_of']

    for sf_dd in sel_flights_dep_dates:
        sel_flights_ss = sel_flights[sel_flights_dep_dates == sf_dd]  # subset of sel. flights

        if len(sel_flights_ss) > 1:
            drift_len_local, drift_local, vol_local_1, vol_local_2, \
                sum_price_tmp = compute_partial_drift_vol( sel_flights_ss['as_of']
                                                         , sel_flights_ss['price']
                                                         , model = model )
            drift += drift_local
            vol_1 += vol_local_1
            vol_2 += vol_local_2
            drift_len += drift_len_local
            sum_price += sum_price_tmp

    if drift_len == 0:
        return None

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

    timestamp_now = datetime.date.today()
    date_diff = np.diff(np.array([(x - timestamp_now).seconds for x in date_l])) / (dcf*86400)

    price_l_diff = np.diff(np.array(price_l))
    price_diff   = price_l_diff/np.array(price_l[:-1]) if model == 'ln' else price_l_diff

    drift_over_sqdate = price_diff/np.sqrt(date_diff)

    return  len(price_l)\
          , np.sum(price_diff/date_diff)\
          , np.sum(drift_over_sqdate**2)\
          , np.sum(drift_over_sqdate)\
          , np.sum(price_l)


def insert_into_db_function(PARAMS):

    # TODO: TO FINISH

    if insert_into_db and (drift_len != 0):  # insert only if there is anything to insert
        with MysqlConnectorEnv(host='localhost') as conn:
            conn.cursor().executemany(
                """INSERT INTO params (as_of, orig, dest, carrier, drift, vol avg_price, reg_id)
                   VALUES( %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s )"""
                , (as_of_date, orig, dest, carrier, drift, vol, avg_price, REG_ID))  # TODO: MISSING HERE
            conn.commit()

        with sqlite3.connect(SQLITE_FILE) as conn_ao:
            conn_ao.execute("INSERT INTO params VALUES ('{0}', '{1}', '{2}', '{3}', '{4}', {5}, {6}, {7}, {8}, '{9}', '{10}')"\
                            .format(as_of_date, orig, dest, '1', carrier, drift, vol, avg_price, dep_season, dep_hour, dep_day))
            conn_ao.commit()


def all_vols_by_airline( carrier : str
                       , insert_into_db    = False
                       , as_of_date        = None
                       , model             = 'n'
                       , correct_drift_vol = False):
    """ Estimates all drift/vol pairs for a particular airline.

    :param carrier: airline, carrier ('UA')
    :param insert_into_db: indicator whether to insert into db.
    :param as_of_date: the date we are running the estimation, if None, today's date/time.
    :param model: log-normal ('ln') or normal('n') model to estimate.
    :param correct_drift_vol: indicator whether to correct drift/volatility w/ large/small values.
    """

    orig_dest_l = run_db_mysql("SELECT DISTINCT orig, dest FROM flights WHERE carrier = '{0}'".format(carrier) )

    for orig, dest in orig_dest_l:  # select a subset of only those
        try:
            flight_vol( orig
                            , dest
                            , carrier
                            , insert_into_db    = insert_into_db
                            , as_of_date        = as_of_date
                            , model             = model
                            , correct_drift_vol = correct_drift_vol)

        except EmptyFlights:  # if fails return empty
            logger.info('No flights between {0} and {1} for carrier {2}'.format(orig, dest, carrier))

        except Exception as e:
            raise e


def all_vols( insert_into_db    = False
            , as_of_date        = None
            , model             = 'n'
            , correct_drift_vol = False ):
    """ Estimates all drift/vol pairs for all airlines.

    """

    for carrier, _ in run_db_mysql("SELECT DISTINCT carrier FROM flights"):
        all_vols_by_airline( carrier           = str(carrier)
                           , insert_into_db    = insert_into_db
                           , as_of_date        = as_of_date
                           , model             = model
                           , correct_drift_vol = correct_drift_vol )


def flight_corr( orig_l     : List[str]
               , dest_l     : List[str]
               , carrier_l  : List[str]
               , dep_date_l : List[datetime.date]
               , dcf        = 365.25 ):
    """
    Compute the correlation between flights in the list

    :param orig_l:    origin list of IATA airports, ['EWR'...]
    :param dest_l:    destination list of IATA airports ['SFO', ...]
    :param carrier_l: list of carrier airlines (IATA codes) ['UA', 'B6'...]
    :returns:
    """

    nb_flights = len(orig_l)  # they should all be the same     
    df = dict()
    
    for flight_nb, orig, dest, carrier, dep_date in zip(range(nb_flights),
                                                        orig_l, dest_l, carrier_l, dep_date_l):
        direct_flights_morning = """
        SELECT * 
        FROM flights 
        WHERE orig = {0} AND dest = {1} AND carrier = '{2}' 
              AND dep_date = '{3}' AND direct_flight = 1
        """.format(orig, dest, carrier, dep_date)

        with sqlite3.connect(SQLITE_FILE) as conn_ao:
            df[flight_nb] = pd.read_sql_query( direct_flights_morning
                                             , conn_ao
                                             , parse_dates = { 'as_of'   : '%Y-%m-%d'
                                                             , 'dep_date': '%Y-%m-%dT%H:%M:%S'
                                                             , 'arr_date': '%Y-%m-%dT%H:%M:%S' })

        # THIS BELOW CAN BE WRITTEN IN A VECTOR FORM - CORRECT CORRECT CORRECT 
        nb_flights = len(df)
        # construct the dates
        price_diff = np.empty(nb_flights-1)
        date_diff  = np.empty(nb_flights-1)
        
        for flight_nb in range(1, nb_flights):
            price_diff[flight_nb-1] = df['price'][flight_nb] - df['price'][flight_nb-1]
            date_diff[flight_nb-1] = (df['as_of'][flight_nb] - df['as_of'][flight_nb-1]).days / dcf

            drift = np.mean(price_diff/date_diff)
            vol = np.std(price_diff/date_diff)


def find_flight_ids( orig     : str
                   , dest     : str
                   , carrier  : str
                   , min_nb_obs = 0
                   , host       = 'localhost' ) -> pd.DataFrame :
    """
    returns the flight ids for a flight from orig to dest on carrier

    :param orig:       IATA code of the origin airport (like 'SFO')
    :param dest:       IATA code of the destination airport (like 'EWR')
    :param carrier:    IATA code of the airline (like 'UA')
    :param min_nb_obs: minimum number of observations, include only those
    :returns:          flight_ids for the conditions specified
    """

    flight_ids_str = "SELECT flight_id FROM flight_ids WHERE orig = '{0}' AND dest='{1}' AND carrier = '{2}'".format(orig, dest, carrier)

    # THIS BELOW IS NOT WORKING 
    if min_nb_obs > 0:
        flight_ids_str += " AND COUNT(flight_id) > {0}".format(min_nb_obs)

    with MysqlConnectorEnv(host=host) as mysql_conn:
        return pd.read_sql_query( flight_ids_str, mysql_conn)
