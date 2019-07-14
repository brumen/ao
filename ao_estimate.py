# estimate the volatility and drift of air option stochastic processes
#

import logging
import datetime
import time
import sqlite3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing      import List, Tuple
from scipy.stats import norm  # quantiles of normal distr. norm.ppf(x)
from tkinter     import Frame, Button, Label

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


def flight_price_get(flight_id : int):
    """
    Returns prices from flight number id and computes drift and vol for
    those prices

    :param flight_id: Flight id to display prices in the database for
    :returns:         tuple of data_frame, (drift, vol) computed for the flight_id
    :rtype:           tuple of DataFrame, (double, double)
    """

    flights_prices_str = """
    SELECT * 
    FROM flights_ord 
    WHERE flight_id = {0}
    ORDER BY as_of
    """.format(flight_id)

    orig_dest_str = """
    SELECT orig, dest, carrier 
    FROM flight_ids 
    WHERE flight_id = {0}
    """.format(flight_id)

    with MysqlConnectorEnv() as m_conn:

        df1 = pd.read_sql_query( flights_prices_str
                               , m_conn
                               , parse_dates = {'as_of': '%Y-%m-%d'})
        df2 = pd.read_sql_query( orig_dest_str
                               , m_conn)  # this is unique

        reg_id = df1['reg_id'][0]  # they are all the same, take first

        orig, dest, carrier = df2['orig'][0], df2['dest'][0], df2['carrier'][0]

        df3 = pd.read_sql_query( "SELECT drift, vol FROM params WHERE orig = '{0}' AND dest = '{1}' AND carrier = '{2}' AND reg_id = '{3}'".format(orig, dest, carrier, reg_id)
                               , m_conn)  # this is unique

        drift, vol = df3['drift'][0], df3['vol'][0]

    return df1, (drift, vol)


def compute_conf_band( x0
                     , ts
                     , drift
                     , vol
                     , quant = 0.9):
    """Constructs mean, confidence bands for log-normal process starting from x0,

    :param ts: time-series
    :param x0: starting value of time series
    :param drift, vol: drift, vol of the process
    :param quant: quantile to which to display bounds 
    :returns:
    """

    mean_ts = x0 + ts * drift
    quant = norm.ppf(0.5 - quant/2.)
    deviat_ts_pos = vol * np.sqrt(ts) * quant
    lower_q_ts = mean_ts + deviat_ts_pos
    upper_q_ts = mean_ts - deviat_ts_pos  # upper and lower are symmetric 

    return mean_ts, lower_q_ts, upper_q_ts


def plot_flight_prices( df1
                      , drift = 100.
                      , vol   = 200. ):
    """
    uses df1 from flight_price_get to plot flight prices

    """
    # compute date diffs
    df1d = df1['as_of'].diff()  # this will express differences in timedelta

    # construct time series in normalized units 
    ts = np.empty(len(df1d))
    ts[0] = 0.
    ts[1:] = np.array([x.total_seconds() / (86400 * DCF)
                       for x in df1d[1:]])  # numerical value of days
    ts = ts.cumsum()

    mean_ts, lower_q_ts, upper_q_ts = compute_conf_band(df1['price'][0], ts, drift, vol)
    
    # print ts, type(ts), fitted_prices, df1['price']
    data_df = np.array([np.array(df1['price']), mean_ts, lower_q_ts, upper_q_ts]).transpose()
    legends = ['price', 'mean', 'q05', 'q95']
    df2 = pd.DataFrame(data=data_df, index=ts, columns=legends)
    ax1 = df2.plot()
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines[:4], legends, loc='best') 
    plt.show(block=False)

    return df2, ax1
    

def plot_from_flight_id(flight_id):
    """
    Plots the graph of actual, mean, quantile prices from flight_id

    :param flight_id: id of the flight plotted
    :type flight_id:  int
    :returns:         plots the flight
    :rtype:           None
    """

    df1, drift_vol = flight_price_get(flight_id)

    if len(df1) > 1:
        drift, vol = drift_vol
        df2,   ax1 = plot_flight_prices(df1, drift=drift)
        return df1, df2, ax1
    else:
        print ("No flights registered for flight_id {0}".format(flight_id))


class ArrayButtons(Frame):

    def __init__( self
                , mat
                , master=None):
        """

        :params mat: matrix of rows, columns of flight numbers, like 1533531
        """
        Frame.__init__(self, master)
        self.pack()
        self.mat = mat   # holding the matrix 

        self.btn = [[0 for y in range(len(mat[x]))] for x in range(len(mat))]

        for x in range(len(mat)):
            for y in range(len(mat[x])):
                self.btn[x][y] = Button( self
                                       , text    = mat[x][y]
                                       , command = lambda x1=x, y1=y: self.display_graph(x, y))
                self.btn[x][y].grid(column=x, row=y)

    def display_graph(self, x1, y1):
        """
        plots the graph TODO: FILL IN HERE

        :param x1: x-index of the matrix to display
        :type x1:  int
        :param y1: y-index of the matrix to display
        :type y1:  int
        """
        plot_from_flight_id(flight_id=self.mat[x1][y1])


class ArrayTextButtons(Frame):
    """
    mat ... matrix of rows, columns 
    if there are more than 30 items, rearrange them in columns of 30
    """

    def __init__(self, text_fid_mat, master=None):
        """
        display flights w/ all the goodies

        :param f_l: flight list in the form [(some_text, flight_nb), ...]
                      some text can be anything you construct 
        """

        Frame.__init__(self, master)
        self.pack()
        self.text_fid_mat = text_fid_mat   # list of (text, fid)
        self.nb_flights = len(text_fid_mat)
        self.nb_columns = self.nb_flights/30 + 1
        self.btn = [[0 for x in range(2*self.nb_columns)] for x in range(self.nb_flights)]
        self.curr_plots = 0
        
        for idx, tf_curr in enumerate(text_fid_mat):
            curr_column = idx/30
            curr_row = idx - curr_column * 30
            
            text_curr, fid_curr = tf_curr
            curr_column_used = 2*curr_column
            self.btn[curr_row][curr_column_used + 1] = Button( self
                                                             , text=fid_curr
                                                             , command = lambda x1=curr_row, y1=curr_column:
                                                                  self.display_graph(x1, y1))
            self.btn[curr_row][curr_column_used] = Label(self, text=text_curr)

            self.btn[curr_row][curr_column_used + 1].grid(column=curr_column_used + 1, row=curr_row)
            self.btn[curr_row][curr_column_used].grid(column=curr_column_used, row=curr_row)

    def display_graph(self, x1, y1):

        if self.curr_plots == 0:
            self.curr_plots = 1

        else:  # close the other one, display new one
            plt.close()

        flight_idx_compute = 30*y1 + x1
        plot_from_flight_id(flight_id=self.text_fid_mat[flight_idx_compute][1])
