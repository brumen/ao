# getting the params from the database
import datetime

from typing import List, Tuple

# air options modules
from ao.ds import convert_datedash_date
import ao.ao_codes as ao_codes  # TODO: CHANGE THIS

from ao.mysql_connector_env import MysqlConnectorEnv

from ao.ao_codes import day_str as all_ranks


def get_drift_vol_from_db( dep_date : datetime.date
                         , orig     : str
                         , dest     : str
                         , carrier  : str
                         , default_drift_vol = (500., 501.)
                         , fwd_value         = None
                         , db_host           = 'localhost' ) -> Tuple[float, float] :
    """ Pulls the drift and vol from database for the selected flight.

    :param dep_date: departure date of the flight (datetime.date(2019, 7, 1) )
    :param orig: IATA code of the origin airport ('EWR')
    :param dest: IATA code of the destination airport ('SFO')
    :param carrier: airline carrier, e.g. 'UA'
    :param default_drift_vol: correct the drift, if negative make it positive 500, or so.
    :param fwd_value: forward value used in case we want to correct the drift. If None, take the original drift.
    :param db_host: database host, e.g. 'localhost'
    """

    with MysqlConnectorEnv(host=db_host) as drift_connection:
        drift_vol_c = drift_connection.cursor()
        drift_vol_c.execute("SELECT as_of, drift, vol, avg_price FROM params WHERE orig= '{0}' AND dest = '{1}' AND carrier='{2}'".format(orig, dest, carrier))
        drift_vol_params = drift_vol_c.fetchall()

    if len(drift_vol_params) == 0:  # nothing in the list
        return default_drift_vol

    # at least one entry, check if there are many, select most important
    # entry in form (datetime.datetime, drift, vol, avg_price)
    closest_date_params = sorted(drift_vol_params, key=lambda drift_vol_param: abs((dep_date - drift_vol_param[0].date()).days))[0]
    _, drift_prelim, vol_prelim, avg_price = closest_date_params

    return correct_drift_vol( drift_prelim
                            , vol_prelim
                            , default_drift_vol
                            , avg_price
                            , fwd_value )


def correct_drift_vol( drift_prelim      : float
                     , vol_prelim        : float
                     , default_drift_vol : Tuple[float, float]
                     , avg_price         : float
                     , fwd_price = None ) -> Tuple[float, float]:
    """ Applies trivial corrections to the drift and vol in case of nonsensical results.
    The scaling factor is fwd_price/avg_price

   :param drift_prelim: preliminary drift
   :param vol_prelim: preliminary volatility
   :param default_drift_vol: tuple of default vol and drift
   :param avg_price: average price for the flight considered
   :param fwd_price: forward price of the flight
   :returns: corrected drift and volatility as described above.
    """

    if (drift_prelim, vol_prelim) == (0., 0.):  # this means there is no price change in the database
        return default_drift_vol

    # first drift, then vol
    if drift_prelim < 0.:
        drift_prelim = default_drift_vol[0]  # take a large default value, always change drift

    if not fwd_price:  # fwd_price == None
        return drift_prelim, vol_prelim

    # correct the drift/vol by rescaling it
    scale_f = fwd_price/avg_price
    return scale_f * drift_prelim, scale_f * vol_prelim


def get_drift_vol_from_db_precise( flight_dep_l : List
                                 , orig         : str
                                 , dest         : str
                                 , carrier      : str
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
    flight_dep_month_l   = [ convert_datedash_date(d_t[0]).month
                             for d_t in flight_dep_time_l ]  # months extracted
    flight_dep_weekday_l = [ ao_codes.get_weekday_ind(convert_datedash_date(d_t[0]).day)
                             for d_t in flight_dep_time_l ]
    # month == integer
    # tod == text 'night'
    # weekday_ind == 'weekday', 'weekend'

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
                    , tod     : str
                    , weekday : str ):
    """ Logic how to select regid which is close to desired one

    :param month: month desired.
    :param tod: time of day, 'morning', 'afternoon', 'evening', 'night'
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

    tod_idx = all_ranks.index(tod)
    tod_ranks = [ all_ranks[tod_idx]
                , all_ranks[(tod_idx+1) % 4]
                , all_ranks[(tod_idx+2) % 4]
                , all_ranks[(tod_idx+3) % 4]]

    weekday_ranks = ['weekday', 'weekend'] if weekday == 'weekday' else ['weekend', 'weekday']

    return month_ranks, tod_ranks, weekday_ranks
