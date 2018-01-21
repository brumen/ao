# estimate the volatility and drift of air option stochastic processes
#

import numpy as np
import pandas as pd
import datetime as dt
import time
import sqlite3
import mysql.connector
import pymysql
import matplotlib.pyplot as plt
from scipy.stats import norm  # quantiles of normal distr. norm.ppf(x)
from Tkinter import Frame, Button, Label

import ds
import ao_db
import ao_codes
from   ao_db               import SQLITE_FILE
from   ao_codes            import DB_HOST, DATABASE, DB_USER
from   mysql_connector_env import MysqlConnectorEnv


class EmptyFlights(Exception):
    pass


def date_in_season(date, season):
    """
    checks whether date is in season
    :param date: given in date format, datetime format or any list related format 
    :param season: either 'summer' or 'winter'
    """
    cnd_outer = type(date) is dt.date or type(date) is dt.datetime
    if season == 'summer':
        if cnd_outer:
            month_cnd = date.month >= 5 and date.month < 10
        else:  # it's a list type object
            month_cnd = [d.month >= 5 and d.month < 10 for d in date]
    else:  # season == winter 
        if cnd_outer: 
            month_cnd = date.month <=4 or date.month >= 10
        else:
            month_cnd = [d.month <= 4 or d.month >= 10 for d in date]

    return month_cnd


def hour_in_section(hour, section):
    """
    checks whether the hour is in specified section
    :param hour: hour in datetime format or a list format 
    :param section: either 'morning', 'afternoon', 'evening', 'night'
    """
    def hour_in_section_atomic(hour_used, sect):
        if sect == 'morning':
            return hour_used >= 6 and hour_used < 11
        elif sect == 'afternoon':
            return hour_used >= 11 and hour_used < 18
        elif sect == 'evening':
            return hour_used >= 18 and hour_used < 23
        else:
            return hour_used >= 23 or hour_used < 6

    if type(hour) is dt.time:
        return hour_in_section_atomic(hour.hour, section)
    else:  # a list, go over
        return [hour_in_section_atomic(h.hour, section) for h in hour]


def regression_check_results():
    """
    checks the updated database versus the old one, stored in params table 
    """ 
    mysql_conn_1 = pymysql.connect(host='localhost', database='ao',
                                   user='brumen', passwd=ao_codes.brumen_mysql_pass)
    mysql_c_1 = mysql_conn_1.cursor()

    old_params_str = """SELECT * from params"""
    df_old = pd.read_sql_query(old_params_str, mysql_conn_1,
                               parse_dates={'as_of': '%Y-%m-%d'})
    #new_params_str = """CALL calibrate_all_final()"""
    #df_new = pd.read_sql_query(new_params_str, mysql_conn_1,
    #                           parse_dates={'as_of': '%Y-%m-%d'})
    # compare the two data frames, no clue how to do that 
    return df_old
    

def compute_partial_drift_vol(date_l, price_l, dcf=365.,
                              model='n', debug_ind=False):
    nb_p = len(price_l)
    date_diff = np.array([x.total_seconds() for x in date_l.diff()[1:]]) / (dcf*86400)
    if model == 'ln':
        price_diff = np.array(price_l.diff()[1:])/np.array(price_l[:-1])
    else:
        price_diff = np.array(price_l.diff()[1:])

    drift_over_date = price_diff/date_diff
    drift_over_sqdate = price_diff/np.sqrt(date_diff)
    drift_local = np.sum(drift_over_date)
    vol_local_1 = np.sum(drift_over_sqdate**2)  # first term in the vol
    vol_local_2 = np.sum(drift_over_sqdate)
    sum_price = np.sum(price_l)
    
    return nb_p, drift_local, vol_local_1, vol_local_2, sum_price


def flight_vol( orig              = 'SFO'
              , dest              = 'EWR'
              , carrier           = 'UA'
              , dcf               = 365.
              , insert_into_db    = False
              , as_of_date        = None
              , use_cache         = False
              , model             = 'n'
              , correct_drift_vol = False):
    """

    """

    lt = time.localtime()
    as_of = str(lt.tm_year) + '-' + str(ds.d2s(lt.tm_mon)) + '-' + str(ds.d2s(lt.tm_mday)) + 'T' + \
            str(ds.d2s(lt.tm_hour)) + ':' + str(ds.d2s(lt.tm_min)) + ':' + str(ds.d2s(lt.tm_sec))
        
    if as_of_date is None:
        as_of_used = as_of
    else:
        as_of_used = as_of_date

    if use_cache:
        db_used = "flights_cache"
        price_field = 'min_price'
    else:
        db_used = "flights"
        price_field = 'price'
        
    direct_flights = """
    SELECT DISTINCT as_of, orig, dest, {0}, month, tod, weekday_ind  
    FROM {1} 
    WHERE orig= '{2}' AND dest = '{3}' AND carrier='{4}' 
    ORDER BY as_of""".format(price_field, db_used, orig, dest, carrier)

    if not use_cache: 
        df1 = pd.read_sql_query(direct_flights, ao_db.conn_ao,
                                parse_dates={ 'as_of'    : '%Y-%m-%d'
                                            , 'dep_date' : '%Y-%m-%dT%H:%M:%S'
                                            , 'arr_date' : '%Y-%m-%dT%H:%M:%S'})
    else:
        df1 = pd.read_sql_query(direct_flights, ao_db.conn_ao,
                                parse_dates={'as_of': '%Y-%m-%dT%H:%M:%S',
                                             'dep_date': '%Y-%m-%dT%H:%M:%S'})
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
                                 , dcf               = dcf
                                 , insert_into_db    = insert_into_db
                                 , as_of_date        = as_of_used
                                 , use_cache         = use_cache
                                 , model             = model
                                 , correct_drift_vol = correct_drift_vol)


def flight_vol_mysql( orig              = 'SFO'
                    , dest              = 'EWR'
                    , carrier           = 'UA'
                    , dcf               = 365.
                    , insert_into_db    = False
                    , as_of_date        = None
                    , model             = 'n'
                    , correct_drift_vol = False):

    lt = time.localtime()
    as_of = str(lt.tm_year) + '-' + str(ds.d2s(lt.tm_mon)) + '-' + str(ds.d2s(lt.tm_mday)) + 'T' + \
            str(ds.d2s(lt.tm_hour)) + ':' + str(ds.d2s(lt.tm_min)) + ':' + str(ds.d2s(lt.tm_sec))
        
    if as_of_date is None:
        as_of_used = as_of
    else:
        as_of_used = as_of_date

    direct_flights = """
    SELECT DISTINCT as_of, orig, dest, price, tod, month, weekday_ind
    FROM flights_ord WHERE orig= '{0}' AND dest = '{1}' AND carrier='{2}' 
    ORDER BY as_of""".format(orig, dest, carrier)

    df1 = pd.read_sql_query( direct_flights
                           , mysql.connector.connect( host     = DB_HOST
                                                    , database = DATABASE
                                                    , user     = DB_USER
                                                    , password = ao_codes.brumen_mysql_pass)
                           , parse_dates={'as_of': '%Y-%m-%d'})
    
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
                                 , dcf               = dcf
                                 , insert_into_db    = insert_into_db
                                 , as_of_date        = as_of_used
                                 , use_cache         = False
                                 , model             = model
                                 , correct_drift_vol = correct_drift_vol)


def flight_vol_mysql_all(dcf               = 365.,
                         insert_into_db    = False,
                         as_of_date        = None,
                         model             = 'n',
                         correct_drift_vol = False):

    
    lt = time.localtime()
    as_of = str(lt.tm_year) + '-' + str(ds.d2s(lt.tm_mon)) + '-' + str(ds.d2s(lt.tm_mday)) + 'T' + \
            str(ds.d2s(lt.tm_hour)) + ':' + str(ds.d2s(lt.tm_min)) + ':' + str(ds.d2s(lt.tm_sec))
        
    if as_of_date is None:
        as_of_used = as_of
    else:
        as_of_used = as_of_date

    first_sql = """
    SELECT f.reg_id reg_id, fid.carrier carrier, fid.orig orig, fid.dest dest, 
           f.flight_id flight_id, f.as_of as_of, f.price price 
    FROM flights_ord f, flight_ids fid
    WHERE f.flight_id = fid.flight_id
    """
    # ??? - GROUP BY f.reg_id, fid.carrier, fid.orig, fid.dest, f.flight_id
    df_first = pd.read_sql_query( first_sql
                                , pymysql.connect( host     = DB_HOST
                                                 , database = DATABASE
                                                 , user     = DB_USER
                                                 , password=ao_codes.brumen_mysql_pass)
                                , parse_dates={'as_of': '%Y-%m-%d'})

    df_group = df_first.groupby(['reg_id', 'carrier', 'orig', 'dest', 'flight_id'])

    # for dep_hour in ['morning', 'afternoon', 'evening', 'night']:
    #     for dep_day in ['weekday', 'weekend']:
    #         for dep_season in range(1,13):
    #            flight_vol_intern(orig, dest, carrier,
    #                              df1, 
    #                              dep_hour=dep_hour, dep_day=dep_day, dep_season=dep_season, 
    #                              c_ao=ao_db.c_ao, conn_ao=ao_db.conn_ao,
    #                              dcf=dcf, insert_into_db=insert_into_db,
    #                              as_of_date=as_of_used,
    #                              use_cache=False,
    #                              model=model,
    #                              correct_drift_vol=correct_drift_vol)


def flight_vol_intern( orig
                     , dest
                     , carrier
                     , df1
                     , dep_hour          = 'morning'
                     , dep_day           = 'weekday'
                     , dep_season        = 10
                     , dcf               = 365.
                     , insert_into_db    = False
                     , as_of_date        = None
                     , use_cache         = False
                     , model             = 'n'
                     , correct_drift_vol = False):
    """
    compute the volatility of particular flight 

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

    if use_cache:
        db_used = "flights_cache"
        price_field = 'min_price'
    else:
        db_used = "flights"
        price_field = 'price'

    
    drift = 0.  # this will be incrementally added 
    drift_len = 0
    vol_1, vol_2 = 0., 0.
    sum_price = 0.

    # filter the dates 
    # filter_used = [(x.month == dep_season) and (x.dayofweek in dof_l) and
    #                (x.time() >= hr_s) and (x.time() < hr_e) 
    #                for x in df1['dep_date']]  # this is really SLOW

    filter_used = [(df1['month'] == dep_season) and (df1['tod'] == dep_hour) and
                   (df1['weekday_ind'] == dep_day)]
    
    # :param dep_hour: departure hour, one of 'morning', 'afternoon', 'evening', 'night'
    # :param dep_day: 'weekday', 'weekend' 
    # :param dep_season: month of departure (1, 2, ... 12)


    
    # alternative way of doing the filter_used
    
    # def dof_cond(ser_df, dof_l):
    #     dof_res = ser_df == dof_l[0]  # np.zeros(len(ser_df), dtype=bool)
    #     for dof in dof_l[1:]:
    #         np.logical_or(dof_res, ser_df == dof)
    #     return dof_res
    # df1_u = df1['dep_date']
    # filter_used_l = [df1_u.dt.month == dep_season,
    #                  #df1_u.dt.dayofweek in dof_l,
    #                  dof_cond(df1_u.dt.dayofweek, dof_l),
    #                  df1_u.dt.time >= hr_s,
    #                  df1_u.dt.time < hr_e]
    # filter_used = np.logical_and.reduce(filter_used_l)


    sel_flights = df1[filter_used].drop_duplicates(subset='as_of')
    sel_flights_dep_dates = sel_flights['dep_date']
    for sf_dd in sel_flights_dep_dates:
        sel_flights_ss = sel_flights[sel_flights_dep_dates == sf_dd]  # subset of sel. flights

        if len(sel_flights_ss) > 1:
            drift_len_local, drift_local, vol_local_1, vol_local_2, \
                sum_price_tmp = compute_partial_drift_vol(sel_flights_ss['as_of'],
                                                          sel_flights_ss[price_field],
                                                          model=model)
            drift += drift_local
            vol_1 += vol_local_1
            vol_2 += vol_local_2
            drift_len += drift_len_local
            sum_price += sum_price_tmp
    
    # order flights by dep_date
    # for dep_date in dep_dates['dep_date']:
    #     # select dates from df1 which have dep_date equal to dep_date
    #     sel_flights_orig = df1[df1['dep_date'] == dep_date]
    #     sel_flights = sel_flights_orig.drop_duplicates(subset='as_of')  # first round selection
    #     # sel_flights = self_flights_fr[sel_flights_fr['id'] == flight_id]
    #     if len(sel_flights) > 1:
    #         # flights are in order
    #         drift_len_local, drift_local, vol_local_1, vol_local_2, \
    #             sum_price_tmp = compute_partial_drift_vol(sel_flights['as_of'],
    #                                                       sel_flights[price_field],
    #                                                       model=model)
    #         drift += drift_local
    #         vol_1 += vol_local_1
    #         vol_2 += vol_local_2
    #         drift_len += drift_len_local
    #         sum_price += sum_price_tmp

    if drift_len == 0:
        return None, None, None, None
    else:
        drift /= drift_len
        vol = np.sqrt((vol_1/drift_len - (vol_2/drift_len)**2))
        avg_price = np.double(sum_price) / drift_len

    # final checks
    all_valid = (drift_len != 0)
    if correct_drift_vol and (drift <= 0):
        drift = 500.  # large value TODO: MAKE THIS VALUE IN THE CONFIG FILE
        
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
                       , dcf               = 365.
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
                      , dcf=dcf
                      , insert_into_db=insert_into_db
                      , as_of_date=as_of_date
                      , use_cache=use_cache
                      , model=model
                      , correct_drift_vol=correct_drift_vol)

        except EmptyFlights:
            vol, drift, nb_samples, avg_price = None, None, None, None

    return 0


def all_vols_by_airline_mysql( carrier           = 'UA'
                             , dcf               = 365.
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
                            , dcf               = dcf
                            , insert_into_db    = insert_into_db
                            , as_of_date        = as_of_date
                            , model             = model
                            , correct_drift_vol = correct_drift_vol)

        except EmptyFlights:
            vol, drift, nb_samples, avg_price = None, None, None, None

    return 0


def all_vols( dcf=365.
            , insert_into_db=False
            , as_of_date=None
            , use_cache=False
            , model='n'
            , correct_drift_vol=False):
    """
    estimates all drift/vol pairs for all airlines 

    :param mp_ind: multi-processing parameter (_DOES NOT WORK_)
    """
    # list of airlines
    if use_cache:
        db_used = 'flights_cache'
    else:
        db_used = 'flights'
    airline_l = ao_db.run_db("""SELECT DISTINCT carrier FROM %s""" % db_used)
    airline_l_str = [str(carrier[0]) for carrier in airline_l]

    for carrier in airline_l_str:
        all_vols_by_airline( carrier           = carrier
                           , dcf               = dcf
                           , insert_into_db    = insert_into_db
                           , as_of_date        = as_of_date
                           , use_cache         = use_cache
                           , model             = model
                           , correct_drift_vol = correct_drift_vol)


def all_vols_mysql( dcf=365.
                  , insert_into_db=False
                  , as_of_date=None
                  , model='n'
                  , correct_drift_vol=False ):
    """
    estimates all drift/vol pairs for all airlines 

    :param mp_ind: multi-processing parameter (_DOES NOT WORK_)
    """
    # list of airlines
    airline_l = ao_db.run_db_mysql("SELECT DISTINCT carrier FROM flights")
    airline_l_str = [str(carrier[0]) for carrier in airline_l]

    for carrier in airline_l_str:
        all_vols_by_airline_mysql( carrier           = carrier
                                 , dcf               = dcf
                                 , insert_into_db    = insert_into_db
                                 , as_of_date        = as_of_date
                                 , model             = model
                                 , correct_drift_vol = correct_drift_vol)


def flight_corr( orig_l         = ['SFO']
               , dest_l         = ['JFK']
               , carrier_l      = ['B6']
               , dep_date_l     = ['2016-10-01T15:10:00']
               , dcf            = 365.
               , insert_into_db = False
               , as_of_date     = '2016-09-11'):
    """
    compute the correlation between flights in the list

    :param dcf: day count factor 
    """

    lt = time.localtime()
    as_of = str(lt.tm_year) + '-' + str(ds.d2s(lt.tm_mon)) + '-' + str(ds.d2s(lt.tm_mday))

    nb_flights = len(orig_l)  # they should all be the same     
    df = dict()
    
    for flight_nb, orig, dest, carrier, dep_date in zip(range(nb_flights),
                                                        orig_l, dest_l, carrier_l, dep_date_l):
        direct_flights_morning = """
        SELECT * from flights WHERE orig= '%s' AND dest = '%s' AND carrier='%s' 
        AND dep_date = '%s' AND direct_flight = 1""" % (orig, dest, carrier, dep_date)

        df[flight_nb] = pd.read_sql_query(direct_flights_morning, ao_db.conn_ao,
                                          parse_dates={'as_of': '%Y-%m-%d',
                                                       'dep_date': '%Y-%m-%dT%H:%M:%S',
                                                       'arr_date': '%Y-%m-%dT%H:%M:%S'})

        # THIS BELOW CAN BE WRITTEN IN A VECTOR FORM - CORRECT CORRECT CORRECT 
        nb_flights = len(df)
        # construct the dates
        price_diff = np.empty(nb_flights-1)
        date_diff = np.empty(nb_flights-1)
        
        for flight_nb in range(1, nb_flights):
            price_diff[flight_nb-1] = df[price_field][flight_nb] - df[price_field][flight_nb-1]
            date_diff[flight_nb-1] = (df['as_of'][flight_nb] - df['as_of'][flight_nb-1]).days / dcf

            drift = np.mean(price_diff/date_diff)
            vol = np.std(price_diff/date_diff)

        # if insert_into_db:
        # c_ao.execute("INSERT INTO params VALUES ('%s', '%s', '%s', '%s', '%s', %s, %s)"
        #              % (as_of_date, orig, dest, dep_date, carrier, drift, vol))
        # conn_ao.commit()
        
    # return drift, vol


def find_flight_ids(orig='SFO', dest='EWR', carrier='UA',
                    min_nb_obs=0):
    """
    returns the flight ids for a flight from 
    :param orig, dest, carrier: self explanatory
    :param min_nb_obs: minimum number of observations, include only those 
    """
    print "Getting flights for:", orig, "to", dest, carrier
    
    flight_ids_str = """
    SELECT flight_id FROM flight_ids WHERE orig = '%s' AND dest='%s' AND carrier = '%s'
    """ % (orig, dest, carrier)

    # THIS BELOW IS NOT WORKING 
    if min_nb_obs > 0:
        flight_ids_str += " AND COUNT(flight_id) "

    df1 = pd.read_sql_query(flight_ids_str, constr_mysql_conn())
    return df1


def flight_price_get(flight_id=218312,
                     drift=100.,
                     dcf=365.):
    """
    find prices from flight number 
    """

    flights_prices_str = """
    SELECT * FROM flights_ord WHERE flight_id = %s ORDER BY as_of
    """ % flight_id
    orig_dest_str = """
    SELECT orig, dest, carrier FROM flight_ids WHERE flight_id = %s""" % flight_id

    with MysqlConnectorEnv() as m_conn:

        df1 = pd.read_sql_query( flights_prices_str
                               , m_conn
                               , parse_dates = {'as_of': '%Y-%m-%d'})
        reg_id = df1['reg_id'][0]  # they are all the same, take first
    
        df2 = pd.read_sql_query(orig_dest_str, m_conn)  # this is unique
        orig, dest, carrier = (df2['orig'][0], df2['dest'][0], df2['carrier'][0])

        params_str = """
        SELECT drift, vol FROM params WHERE orig = '%s' AND dest = '%s' AND carrier = '%s' AND reg_id = '%s'
        """ % (orig, dest, carrier, reg_id)

        df3 = pd.read_sql_query(params_str, m_conn)  # this is unique
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
                      , drift=100.
                      , vol=200.
                      , dcf=365.):
    """
    uses df1 from flight_price_get to plot flight prices

    """
    # compute date diffs
    df1d = df1['as_of'].diff()  # this will express differences in timedelta

    # construct time series in normalized units 
    ts = np.empty(len(df1d))
    ts[0] = 0.
    ts[1:] = np.array([x.total_seconds() / (86400 * dcf) for x in df1d[1:]])  # numerical value of days
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
    

def plot_from_flight_id(flight_id=218312):
    """
    plots the graph of actual, mean, quantile prices from flight_id

    """
    df1, drift_vol = flight_price_get(flight_id)

    if len(df1) > 1:
        drift, vol = drift_vol
        df2,   ax1 = plot_flight_prices(df1, drift=drift)
        return df1, df2, ax1
    else:
        print "No flights registered for flight_id", flight_id


def get_flights_nbs(carrier):
    query_str = """
    SELECT fid.orig orig, fid.dest dest, f.flight_id flight_id,
    COUNT(f.as_of) nb_obs_raw  /* , counti(td(f.as_of)) nb_obs */
    FROM flight_ids fid INNER JOIN flights_ord f
    ON f.flight_id = fid.flight_id WHERE fid.carrier = '%s'
    GROUP BY fid.orig, fid.dest, f.flight_id""" % carrier

    query_str_short = """
    CALL count_flights_by_carrier('%s')""" % carrier

    with MysqlConnectorEnv() as conn:
        df = pd.read_sql_query(query_str_short, conn)

    return df


def display_flights( orig='SFO'
                   , dest='EWR'
                   , carrier='UA'
                   , min_nb_obs = 0):
    """
    TODO: FINISH THIS

    """
    df1 = find_flight_ids( orig       = orig
                         , dest       = dest
                         , carrier    = carrier
                         , min_nb_obs = min_nb_obs)

    fids = np.array(df1['flight_id'][0:100]).reshape((10,10))
    ArrayButtons(fids)


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
