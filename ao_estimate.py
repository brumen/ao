# estimate the volatility and drift of air option stochastic processes
#

import datetime
import time
import sqlite3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm  # quantiles of normal distr. norm.ppf(x)
from tkinter import Frame, Button, Label

import ds
import ao_db
from   ao_db               import SQLITE_FILE
from   ao_codes            import LARGE_DRIFT, DCF
from   mysql_connector_env import MysqlConnectorEnv


class EmptyFlights(Exception):
    pass


def date_in_season(date : datetime.date, season : str):
    """
    checks whether date is in season
    :param date: given in date format, datetime format or any list related format 
    :param season: either 'summer' or 'winter'
    """

    cnd_outer = type(date) is datetime.date or type(date) is datetime.datetime

    if season == 'summer':
        if cnd_outer:
            month_cnd = 5 <= date.month < 10
        else:  # it's a list type object
            month_cnd = [ 5 <= d.month < 10 for d in date]

    else:  # season == winter
        if cnd_outer: 
            month_cnd = date.month <= 4 or date.month >= 10
        else:
            month_cnd = [d.month <= 4 or d.month >= 10 for d in date]

    return month_cnd


def hour_in_section_atomic(hour_used, sect):
    """
    If ?? TODO: FINISH HERE

    """

    if sect == 'morning':
        return 6 <= hour_used < 11
    elif sect == 'afternoon':
        return 11 <= hour_used < 18
    elif sect == 'evening':
        return 18 <= hour_used < 23
    else:
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
    else:  # a list, go over
        return [hour_in_section_atomic(h.hour, section) for h in hour]


def compute_partial_drift_vol( date_l
                             , price_l
                             , model = 'n' ):
    """
    Compute the drift and volatility of the normal/lognormal model.

    :param date_l:  list of dates
    :type date_l:   list of datetime.date
    :param price_l: list of prices at those dates
    :type price_l:  list of double
    :param dcf:     day-count factor
    :type dcf:      double
    :param model:   model selected: 'n' for normal, 'ln' for log-normal
    :type model:    str
    :returns:       tuple of number of prices, drift computed,
                       vol computed E(X**2), vol_computed part E(X)
                       sum of prices
    """
    nb_p = len(price_l)
    date_diff = np.array([x.total_seconds()
                          for x in date_l.diff()[1:]]) / (DCF*86400)

    if model == 'ln':
        price_diff = np.array(price_l.diff()[1:])/np.array(price_l[:-1])
    else:
        price_diff = np.array(price_l.diff()[1:])

    drift_over_date   = price_diff/date_diff
    drift_over_sqdate = price_diff/np.sqrt(date_diff)
    drift_local       = np.sum(drift_over_date)
    vol_local_1       = np.sum(drift_over_sqdate**2)  # first term in the vol
    vol_local_2       = np.sum(drift_over_sqdate)
    sum_price         = np.sum(price_l)
    
    return nb_p, drift_local, vol_local_1, vol_local_2, sum_price


def flight_vol( orig              = 'SFO'
              , dest              = 'EWR'
              , carrier           = 'UA'
              , insert_into_db    = False
              , as_of_date        = None
              , use_cache         = False
              , model             = 'n'
              , correct_drift_vol = False):
    """
    Gets the flight for the following parameters.

    :param orig:    IATA code of the origin airport
    :type orig:     str
    :param dest:    IATA code of the dest. airport
    :type dest:     str
    :param carrier: IATA code of the carrier airline
    """

    lt = time.localtime()
    as_of = '-'.join([str(lt.tm_year), str(ds.d2s(lt.tm_mon)), str(ds.d2s(lt.tm_mday))])\
            + 'T' + \
            ':'.join([str(ds.d2s(lt.tm_hour)), str(ds.d2s(lt.tm_min)), str(ds.d2s(lt.tm_sec))])
        
    if as_of_date is None:
        as_of_used = as_of
    else:
        as_of_used = as_of_date

    if use_cache:
        db_used          = "flights_cache"
        price_field      = 'min_price'
        parse_dates_used = { 'as_of'   : '%Y-%m-%dT%H:%M:%S'
                           , 'dep_date': '%Y-%m-%dT%H:%M:%S' }
    else:
        db_used          = "flights"
        price_field      = 'price'
        parse_dates_used = { 'as_of'   : '%Y-%m-%d'
                           , 'dep_date': '%Y-%m-%dT%H:%M:%S'
                           , 'arr_date': '%Y-%m-%dT%H:%M:%S' }
        
    direct_flights = """
    SELECT DISTINCT as_of, orig, dest, {0}, month, tod, weekday_ind  
    FROM {1} 
    WHERE orig= '{2}' AND dest = '{3}' AND carrier='{4}' 
    ORDER BY as_of""".format(price_field, db_used, orig, dest, carrier)

    with sqlite3.connect(SQLITE_FILE) as conn_ao:

        df1 = pd.read_sql_query( direct_flights
                               , conn_ao
                               , parse_dates = parse_dates_used)

    if len(df1) == 0:  # empty frame
        raise EmptyFlights('No flights found')

    for dep_hour in ['morning', 'afternoon', 'evening', 'night']:
        for dep_day in ['weekday', 'weekend']:
            for dep_season in range(1, 13):

                flight_vol_intern( orig
                                 , dest
                                 , carrier
                                 , df1
                                 , dep_hour          = dep_hour
                                 , dep_day           = dep_day
                                 , dep_season        = dep_season
                                 , insert_into_db    = insert_into_db
                                 , as_of_date        = as_of_used
                                 , use_cache         = use_cache
                                 , model             = model
                                 , correct_drift_vol = correct_drift_vol)


def flight_vol_mysql( orig
                    , dest
                    , carrier
                    , as_of_date=None
                    , insert_into_db    = False
                    , model             = 'n'
                    , correct_drift_vol = False):

    lt = time.localtime()

    as_of = as_of_date if as_of_date is not None else \
        str(lt.tm_year) + '-' + str(ds.d2s(lt.tm_mon)) + '-' + str(ds.d2s(lt.tm_mday)) + 'T' + \
        str(ds.d2s(lt.tm_hour)) + ':' + str(ds.d2s(lt.tm_min)) + ':' + str(ds.d2s(lt.tm_sec))

    direct_flights = """
    SELECT DISTINCT as_of, orig, dest, price, tod, month, weekday_ind
    FROM flights_ord 
    WHERE orig= '{0}' AND dest = '{1}' AND carrier='{2}' 
    ORDER BY as_of""".format(orig, dest, carrier)

    with MysqlConnectorEnv as conn_mysql:
        df1 = pd.read_sql_query( direct_flights
                               , conn_mysql
                               , parse_dates = {'as_of': '%Y-%m-%d'})
    
    if len(df1) == 0:  # empty frame
        raise EmptyFlights('No flights found')

    for dep_hour in ['morning', 'afternoon', 'evening', 'night']:
        for dep_day in ['weekday', 'weekend']:
            for dep_season in range(1,13):
                flight_vol_intern( orig
                                 , dest
                                 , carrier
                                 , df1
                                 , dep_hour          = dep_hour
                                 , dep_day           = dep_day
                                 , dep_season        = dep_season
                                 , insert_into_db    = insert_into_db
                                 , as_of_date        = as_of
                                 , use_cache         = False
                                 , model             = model
                                 , correct_drift_vol = correct_drift_vol )


def flight_vol_intern( orig : str
                     , dest : str
                     , carrier : str
                     , df1
                     , dep_hour          = 'morning'
                     , dep_day           = 'weekday'
                     , dep_season        = 10
                     , insert_into_db    = False
                     , as_of_date        = None
                     , use_cache         = False
                     , model             = 'n'
                     , correct_drift_vol = False):
    """
    compute the volatility of particular flight between orig, dest and carrier

    :param orig: IATA code of the originating airport (like 'EWR')
    :param dest: IATA code of the dest. airport
    :param carrier: IATA code of the carrier airline ('UA')
    :param df1: data frame related to flights, has to be in the form given in flight_vol
    :param dep_hour: departure hour, one of 'morning', 'afternoon', 'evening', 'night'
    :param dep_day: 'weekday', 'weekend' 
    :param dep_season: month of departure (1, 2, ... 12)
    :param as_of_date: which date to use, if None; use the current date
    :param dcf: day count factor 
    :param use_cache: using the cached database 
    :param correct_drift_vol: correct drift and vol: drift to 500. if negative 
    computes the flight volatility and drift 
    """

    price_field = 'min_price' if use_cache else 'price'

    drift        = 0.  # this will be incrementally added
    drift_len    = 0
    vol_1, vol_2 = 0., 0.
    sum_price    = 0.

    filter_used = [(df1['month'] == dep_season) and (df1['tod'] == dep_hour) and
                   (df1['weekday_ind'] == dep_day)]
    
    sel_flights           = df1[filter_used].drop_duplicates(subset='as_of')
    sel_flights_dep_dates = sel_flights['dep_date']

    for sf_dd in sel_flights_dep_dates:
        sel_flights_ss = sel_flights[sel_flights_dep_dates == sf_dd]  # subset of sel. flights

        if len(sel_flights_ss) > 1:
            drift_len_local, drift_local, vol_local_1, vol_local_2, \
                sum_price_tmp = compute_partial_drift_vol( sel_flights_ss['as_of']
                                                         , sel_flights_ss[price_field]
                                                         , model = model )
            drift += drift_local
            vol_1 += vol_local_1
            vol_2 += vol_local_2
            drift_len += drift_len_local
            sum_price += sum_price_tmp

    if drift_len == 0:
        return None, None, None, None
    else:
        drift    /= drift_len
        vol       = np.sqrt((vol_1/drift_len - (vol_2/drift_len)**2))
        avg_price = np.double(sum_price) / drift_len

    # final checks
    all_valid = (drift_len != 0)
    if correct_drift_vol and (drift <= 0):
        drift = LARGE_DRIFT  # large value
        
    if insert_into_db and all_valid:  # insert only if all valid 
        # TODO: CHECK IF THIS IS OK HERE
        with sqlite3.connect(SQLITE_FILE) as conn_ao:
            conn_ao.execute("INSERT INTO params VALUES ('%s', '%s', '%s', '%s', '%s', %s, %s, %s, %s, '%s', '%s')"
                     % (as_of_date, orig, dest, '1', carrier, drift, vol, avg_price, dep_season, dep_hour, dep_day))
            conn_ao.commit()
            
    return drift, vol, drift_len, avg_price


def get_carriers_cache():
    """
    gets different carriers from the cache database
    """
    orig_dest_l = ao_db.run_db("""SELECT DISTINCT carrier FROM flights_cache""")
    return orig_dest_l


def all_vols_by_airline(carrier            = 'UA'
                       , insert_into_db    = False
                       , as_of_date        = None
                       , use_cache         = False
                       , model             = 'n'
                       , correct_drift_vol = False):
    """
    estimates all drift/vol pairs for a particular airline

    """
    # list of origins and destinations
    if use_cache:
        db_used = 'flights_cache'
    else:
        db_used = 'flights'
    orig_dest_l = ao_db.run_db("""SELECT DISTINCT orig, dest FROM %s 
    WHERE carrier = '%s'""" % (db_used, carrier))
    for orig, dest in orig_dest_l:
        # select a subset of only those
        try: 
            flight_vol( orig
                      , dest
                      , carrier
                      , insert_into_db    = insert_into_db
                      , as_of_date        = as_of_date
                      , use_cache         = use_cache
                      , model             = model
                      , correct_drift_vol = correct_drift_vol)

        except EmptyFlights:
            vol, drift, nb_samples, avg_price = None, None, None, None


def all_vols_by_airline_mysql( carrier           = 'UA'
                             , insert_into_db    = False
                             , as_of_date        = None
                             , model             = 'n'
                             , correct_drift_vol = False):
    """
    estimates all drift/vol pairs for a particular airline

    """
    # list of origins and destinations
    orig_dest_l = ao_db.run_db_mysql( """SELECT DISTINCT orig, dest FROM flights 
                                         WHERE carrier = '{0}'""".format(carrier) )

    for orig, dest in orig_dest_l:
        # select a subset of only those
        try: 
            flight_vol_mysql( orig
                            , dest
                            , carrier
                            , insert_into_db    = insert_into_db
                            , as_of_date        = as_of_date
                            , model             = model
                            , correct_drift_vol = correct_drift_vol)

        except EmptyFlights:
            vol, drift, nb_samples, avg_price = None, None, None, None

    return 0


def all_vols( insert_into_db    = False
            , as_of_date        = None
            , use_cache         = False
            , model             = 'n'
            , correct_drift_vol = False):
    """
    estimates all drift/vol pairs for all airlines 

    """
    # list of airlines
    airline_l = ao_db.run_db("SELECT DISTINCT carrier FROM {0}"\
                             .format('flights_cache' if use_cache else 'flights'))
    airline_l_str = map(lambda x: str(x[0]), airline_l)

    for carrier in airline_l_str:
        all_vols_by_airline( carrier           = carrier
                           , insert_into_db    = insert_into_db
                           , as_of_date        = as_of_date
                           , use_cache         = use_cache
                           , model             = model
                           , correct_drift_vol = correct_drift_vol)


def all_vols_mysql( insert_into_db    = False
                  , as_of_date        = None
                  , model             = 'n'
                  , correct_drift_vol = False ):
    """
    estimates all drift/vol pairs for all airlines 

    """
    # list of airlines

    for carrier in map( lambda carrier: str(carrier[0])
                      , ao_db.run_db_mysql("SELECT DISTINCT carrier FROM flights")):

        all_vols_by_airline_mysql( carrier           = carrier
                                 , insert_into_db    = insert_into_db
                                 , as_of_date        = as_of_date
                                 , model             = model
                                 , correct_drift_vol = correct_drift_vol)


def flight_corr( orig_l
               , dest_l
               , carrier_l
               , dep_date_l ):
    """
    Compute the correlation between flights in the list

    :param orig_l:    origin list of IATA airports
    :type orig_l:     list of str ['EWR'...]
    :param dest_l:    destination list of IATA airports
    :type dest_l:     list of str ['SFO', ...]
    :param carrier_l: list of carrier airlines (IATA codes)
    :type carrier_l:  list of ['UA', 'B6'...]
    :returns:
    """

    lt = time.localtime()
    # as_of = str(lt.tm_year) + '-' + str(ds.d2s(lt.tm_mon)) + '-' + str(ds.d2s(lt.tm_mday))
    price_field = 'price'  # TODO: THIS HAS TO BE BETTER INTEGRATED

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
            price_diff[flight_nb-1] = df[price_field][flight_nb] - df[price_field][flight_nb-1]
            date_diff[flight_nb-1] = (df['as_of'][flight_nb] - df['as_of'][flight_nb-1]).days / dcf

            drift = np.mean(price_diff/date_diff)
            vol = np.std(price_diff/date_diff)


def find_flight_ids( orig
                   , dest
                   , carrier
                   , min_nb_obs = 0):
    """
    returns the flight ids for a flight from orig to dest on carrier

    :param orig:       IATA code of the origin airport (like 'SFO')
    :type orig:        str
    :param dest:       IATA code of the destination airport (like 'EWR')
    :type dest:        str
    :param carrier:    IATA code of the airline (like 'UA')
    :type carrier:     str
    :param min_nb_obs: minimum number of observations, include only those
    :type min_nb_obs:  int > 0
    :returns:          flight_ids for the conditions specified
    :rtype:            DataFrame
    """

    flight_ids_str = """
    SELECT flight_id 
    FROM flight_ids 
    WHERE orig = '{0}' AND dest='{1}' AND carrier = '{2}'
    """.format(orig, dest, carrier)

    # THIS BELOW IS NOT WORKING 
    if min_nb_obs > 0:
        flight_ids_str += " AND COUNT(flight_id) > {0}".format(min_nb_obs)

    with MysqlConnectorEnv() as mysql_conn:
        return pd.read_sql_query( flight_ids_str
                                , mysql_conn)


def flight_price_get(flight_id):
    """
    Returns prices from flight number id and computes drift and vol for
    those prices

    :param flight_id: Flight id to display prices in the database for
    :type flight_id:  int
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

    params_str = """
    SELECT drift, vol 
    FROM params 
    WHERE orig = '{0}' AND dest = '{1}' AND carrier = '{2}' AND reg_id = '{3}'
    """

    with MysqlConnectorEnv() as m_conn:

        df1 = pd.read_sql_query( flights_prices_str
                               , m_conn
                               , parse_dates = {'as_of': '%Y-%m-%d'})
        df2 = pd.read_sql_query( orig_dest_str
                               , m_conn)  # this is unique

        reg_id = df1['reg_id'][0]  # they are all the same, take first

        orig, dest, carrier = df2['orig'][0], df2['dest'][0], df2['carrier'][0]

        df3 = pd.read_sql_query( params_str.format(orig, dest, carrier, reg_id)
                               , m_conn)  # this is unique
        drift, vol = df3['drift'][0], df3['vol'][0]

    return df1, (drift, vol)


def compute_conf_band( x0
                     , ts
                     , drift
                     , vol
                     , quant = 0.9):
    """
    Constructs mean, confidence bands for log-normal process starting from x0,

    :param ts: time-series
    :param x0: starting value of time series
    :param drift, vol: drift, vol of the process
    :param quant: quantile to which to display bounds 
    :returns:
    :rtype:       TODO
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
