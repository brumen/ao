# getting the params from the database
import numpy as np
import datetime

# air options modules
import ds
import ao_codes
import ao_db


def get_drift_vol_from_db( dep_date : datetime.date
                         , orig     : str
                         , dest     : str
                         , carrier  : str
                         , default_drift_vol = (500., 501.)
                         , fwd_value         = None):
    """ Pulls the drift and vol from database for the selected flight

    :param dep_date: departure date of the flight (datetime.date(2019, 7, 1) )
    :param orig: IATA code of the origin airport ('EWR')
    :param dest: IATA code of the destination airport ('SFO')
    :param carrier: airline carrier, e.g. 'UA'
    :param correct_drift: correct the drift, if negative make it positive 500, or so.
    :param fwd_value: forward value used in case we want to correct the drift. If None, take the original drift.
    """

    res = ao_db.run_db_mysql("SELECT drift, vol, avg_price FROM params WHERE orig= '{0}' AND dest = '{1}' AND carrier='{2}'".format(orig, dest, carrier))

    if len(res) == 0:  # nothing in the list
        return default_drift_vol

    # at least one entry, check if there are many, select most important
    # entry in form ('date time', drift, vol, avg_price)
    res_date, drift_prelim, vol_prelim, avg_price = select_closest_date(dep_date, res)  # return in the same format
    # final correction in case desired
    return correct_drift_vol( drift_prelim
                            , vol_prelim
                            , default_drift_vol
                            , fwd_value
                            , avg_price)


def select_closest_date(date_desired, date_l):
    """
    Selects the closest date to date_desired in the list date_l.

    :param date_desired: date that you want closest to
    :type date_desired:
    :param date_l:       list of dates that you want to select closest to
    :type date_l:        list of dates
    """

    closest_elt = date_l[0]  # this always exists
    closest_elt_dt = ds.convert_datedash_date(closest_elt[0].split(' ')[0])
    date_desired_dt = ds.convert_datedash_date(date_desired)
    for entry in date_l:
        date_curr_dt = ds.convert_datedash_date(entry[0].split(' ')[0])
        prev_close_distance = abs((date_curr_dt - closest_elt_dt).days)
        new_close_distance = abs((date_curr_dt - date_desired_dt).days)
        if new_close_distance  < prev_close_distance:
            closest_elt = entry
            closest_elt_dt = ds.convert_datedash_date(closest_elt[0].split(' ')[0])

    return closest_elt


def correct_drift_vol( drift_prelim
                     , vol_prelim
                     , default_drift_vol
                     , avg_price
                     , fwd_price = None ):
    """ Applies trivial corrections to the drift and vol in case of nonsensical results.
    The scaling factor is fwd_price/avg_price

   :param drift_prelim: preliminary drift

    """

    if (drift_prelim, vol_prelim) == (0., 0.):  # this means there is no price change in the database
        return default_drift_vol

    # first drift, then vol
    if drift_prelim < 0.:
        drift_prelim = default_drift_vol[0]  # take a large default value, always change drift

    if not fwd_price:  # fwd_price == None
        return drift_prelim, vol_prelim

    # correct the drift/vol by rescaling it
    scale_f = np.double(fwd_price)/avg_price
    return scale_f * drift_prelim, scale_f * vol_prelim


def get_drift_vol_from_db_precise( flight_dep_l
                                 , orig : str
                                 , dest : str
                                 , carrier
                                 , default_drift_vol = (500., 501.)
                                 , correct_drift     = True
                                 , fwd_value         = None
                                 , host_db           = 'localhost' ):
    """
    pulls the drift and vol from database for the selected flight,
    with more information than before

    :param flight_dep_l:  list of flights dep. dates & times in the form 2017-04-06T06:00:00
                             they all share the same date, namely dep_date; so this is not important
    :type flight_dep_l:
    :param orig:          IATA code of the origin airport
    :param dest:          IATA code of the dest airport (i.e. 'EWR')
    :param correct_drift: correct the drift, if negative make it positive 500, or so.
    :type correct_drift:  bool
    :param fwd_value:     forward value used in case correct_drift == True
    :type fwd_value:      double
    :returns:
    :rtype:
    """

    flight_dep_time_l    = [ x.split('T')
                             for x in flight_dep_l ]  # list of dep. dates/times in form ('2017-06-06', '06:00:00')
    flight_dep_tod_l     = [ ao_codes.get_tod(x[1])
                             for x in flight_dep_time_l ]  # list of 'morning' for all flights desired
    flight_dep_month_l   = [ ds.convert_datedash_date(d_t[0]).month
                             for d_t in flight_dep_time_l ]  # months extracted
    flight_dep_weekday_l = [ ao_codes.get_weekday_ind(ds.convert_datedash_date(d_t[0]).day)
                             for d_t in flight_dep_time_l ]
    # month == integer
    # tod == text 'night'
    # weekday_ind == 'weekday', 'weekend'
    from mysql_connector_env import MysqlConnectorEnv

    with MysqlConnectorEnv(host=host_db) as connection:
        mysql_conn_c = connection.cursor()

        selected_drift_vol = """
        SELECT drift, vol, avg_price 
        FROM params 
        WHERE carrier = '{0}' AND orig = '{1}' AND dest = '{2}' AND reg_id = {3}"""


        drift_vol_close_str = """
        SELECT drift, vol, avg_price 
        FROM params 
        WHERE orig = '{0}' AND dest = '{1}' AND carrier = '{2}' AND 
              reg_id IN (SELECT reg_id FROM reg_ids_temp)
        ORDER BY FIELD(reg_id, @reg_id_ord)"""

        drift_vol_l = []
        for dep_date_month, dep_date_tod, dep_date_weekday in zip(flight_dep_month_l,
                                                                  flight_dep_tod_l,
                                                                  flight_dep_weekday_l):

            mysql_conn_c.execute("SELECT reg_id FROM reg_ids WHERE month = {0} AND tod = '{1}' AND weekday_ind = '{2}'".format( dep_date_month
                                                                                                                              , dep_date_tod
                                                                                                                              , dep_date_weekday) )

            reg_ids = mysql_conn_c.fetchone()
            assert len(reg_ids) == 0, 'More reg_ids obtained than there should be'  # TODO: FIX THIS
            reg_id = int(reg_ids[0])
            # first the general query, then the specific one
            mysql_conn_c.execute(selected_drift_vol.format( carrier
                                                          , orig
                                                          , dest
                                                          , reg_id))
            drift_vol_avgp_raw = mysql_conn_c.fetchone()  # there is at most 1

            if drift_vol_avgp_raw is None:  # nothing in the list
                # check if there is anything close to this,
                # reg_ids_similar is a list of reg_ids which are possible
                month_ranks, tod_ranks, weekday_ranks = find_close_regid( dep_date_month
                                                                        , dep_date_tod
                                                                        , dep_date_weekday)
                month_ranks_str   = ','.join([ str(x)
                                               for x in month_ranks ])  # making strings out of these lists
                tod_ranks_str     = ','.join([ "'" + x + "'"
                                               for x in tod_ranks ])
                weekday_ranks_str = ','.join([ "'" + x + "'"
                                               for x in weekday_ranks ])

                mysql_conn_c.execute("""DELETE FROM reg_ids_temp; 
                                        INSERT INTO reg_ids_temp
                                        SELECT reg_id 
                                        FROM reg_ids 
                                        WHERE month IN ({0}) AND weekday_ind = '{1}'
                                        ORDER BY ABS(month - {2}) ASC, FIELD(tod, {3}) ASC, FIELD(weekday_ind, {4}) ASC;
    
                                        SET @reg_id_ord = (SELECT GROUP_CONCAT(reg_id) FROM reg_ids_temp);""".format( month_ranks_str
                                                                                                                    , dep_date_weekday
                                                                                                                    , dep_date_month
                                                                                                                    , tod_ranks_str
                                                                                                                    , weekday_ranks_str)
                                    , multi = True)

                mysql_conn_c.execute(drift_vol_close_str.format( orig
                                                               , dest
                                                               , carrier))
                closest_drift_vol_res = mysql_conn_c.fetchall()

                if not closest_drift_vol_res:  # empty
                    drift_vol_l.append(default_drift_vol)

                else:  # correct the drift
                    drift_prelim, vol_prelim, avg_price = closest_drift_vol_res[0]  # take the first
                    drift_vol_corr = correct_drift_vol( drift_prelim
                                                      , vol_prelim
                                                      , default_drift_vol
                                                      , correct_drift
                                                      , fwd_value
                                                      , avg_price)
                    drift_vol_l.append(drift_vol_corr)

            else:  # drift... has 1 entry
                drift_prelim, vol_prelim, avg_price = drift_vol_avgp_raw
                drift_vol_corr = correct_drift_vol( drift_prelim
                                                  , vol_prelim
                                                  , default_drift_vol
                                                  , correct_drift
                                                  , fwd_value
                                                  , avg_price)
                drift_vol_l.append(drift_vol_corr)
            
    return drift_vol_l


def find_close_regid( month   : int
                    , tod
                    , weekday : str ):
    """
    logic how to select regid which is close to desired one

    :param month: month
    :param tod: time of day
    :type tod:
    :param weekday: weekday, can be either 'weekday' or 'weekend'
    :returns:
    :rtype:
    """

    # months allowed +/- 2
    def rot_month(m, k):
        # rotational month
        return ((m - 1) + k) % 12 + 1

    # months considered
    month_ranks = [ month
                  , rot_month(month, 1)
                  , rot_month(month, -1)
                  , rot_month(month, 2)
                  , rot_month(month, -2)]

    # tod ranking
    from ao_codes import day_str as all_ranks
    tod_idx = all_ranks.index(tod)
    tod_ranks = [ all_ranks[tod_idx]
                , all_ranks[(tod_idx+1) % 4]
                , all_ranks[(tod_idx+2) % 4]
                , all_ranks[(tod_idx+3) % 4]]

    weekday_ranks = ['weekday', 'weekend'] if weekday == 'weekday' else ['weekend', 'weekday']

    return month_ranks, tod_ranks, weekday_ranks
