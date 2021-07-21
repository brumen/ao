""" getting the params from the database
"""

import datetime

from typing import List, Tuple, Optional
from sqlalchemy.orm.session import Session

from ao.ao_codes            import day_str as all_ranks, get_tod, get_weekday_ind
from ao.flight              import create_session, AOParam, AORegIds


def get_drift_vol_from_db( dep_date : datetime.date
                         , orig     : str
                         , dest     : str
                         , carrier  : str
                         , default_drift_vol = (500., 501.)
                         , fwd_value         = None
                         , session           = None ) -> Tuple[float, float] :
    """ Pulls the drift and vol from database for the selected flight.

    :param dep_date: departure date of the flight (datetime.date(2019, 7, 1) )
    :param orig: IATA code of the origin airport ('EWR')
    :param dest: IATA code of the destination airport ('SFO')
    :param carrier: airline carrier, e.g. 'UA'
    :param default_drift_vol: correct the drift, if negative make it positive 500, or so.
    :param fwd_value: forward value used in case we want to correct the drift. If None, take the original drift.
    :param session: sqlalchemy session used for param retrieval.
    """

    session_used = create_session() if session is None else session

    drift_vol_c = session_used.query(AOParam).filter_by(orig=orig, dest=dest, carrier=carrier).all()
    drift_vol_params = [ (ao_param.as_of, ao_param.drift, ao_param.vol, ao_param.avg_price)
                        for ao_param in drift_vol_c]

    if not drift_vol_params:
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
                     , fwd_price         : Optional[float] = None ) -> Tuple[float, float]:
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


def find_close_regid( month   : int
                    , tod     : str
                    , weekday : str ) -> Tuple[ Tuple[int, int, int, int, int]
                                              , Tuple[str, str, str, str]
                                              , Tuple[str, str] ]:

    """ Logic how to select regid which is close to desired one

    :param month: month desired.
    :param tod: time of day, 'morning', 'afternoon', 'evening', 'night'
    :param weekday: weekday, can be either 'weekday' or 'weekend'
    :returns: information about reg_ids that is closest.
    """

    # rotational month, months allowed +/- 2
    rot_month = lambda m, k: ((m - 1) + k) % 12 + 1

    # months considered
    month_ranks = ( month
                  , rot_month(month, 1)
                  , rot_month(month, -1)
                  , rot_month(month, 2)
                  , rot_month(month, -2) )

    tod_idx = all_ranks.index(tod)
    tod_ranks = ( all_ranks[tod_idx]
                , all_ranks[(tod_idx+1) % 4]
                , all_ranks[(tod_idx+2) % 4]
                , all_ranks[(tod_idx+3) % 4] )

    weekday_ranks = ('weekday', 'weekend') if weekday == 'weekday' else ('weekend', 'weekday')

    return month_ranks, tod_ranks, weekday_ranks


def get_drift_vol_from_db_precise( flight_dep   : List[Tuple[datetime.date, datetime.time]]
                                 , orig         : str
                                 , dest         : str
                                 , carrier      : str
                                 , default_drift_vol = (500., 501.)
                                 , correct_drift : bool = True
                                 , fwd_value     : Optional[float]   = None
                                 , session       : Optional[Session] = None ):
    """
    pulls the drift and vol from database for the selected flight,
    with more information than before

    :param flight_dep: list of flights dep. dates & times in the form of a tuple (2017-04-06, 06:00:00)
                       they all share the same date, namely dep_date; so this is not important
    :param orig: IATA code of the origin airport
    :param dest: IATA code of the dest airport (i.e. 'EWR')
    :param carrier: IATA carrier code.
    :param default_drift_vol: default drift and volatility
    :param correct_drift: correct the drift, if negative make it positive 500, or so.
    :param fwd_value:     forward value used in case correct_drift == True
    :param session: session used for interacting w/ the db
    :returns:
    """

    session_used = session if session else create_session()

    # selected_drift_vol = """
    # SELECT drift, vol, avg_price
    # FROM params
    # WHERE carrier = '{0}' AND orig = '{1}' AND dest = '{2}' AND reg_id = {3}"""
    selected_drift_vol = session_used.query(AOParam).filter_by(carrier=carrier, orig=orig, dest=dest, )

    drift_vol_l = []
    for dep_date, dep_time in flight_dep:

        dep_date_month = dep_date.month
        dep_date_tod   = get_tod(dep_time)  # time of day, like 'night'
        dep_date_weekday = get_weekday_ind(dep_date.day)  # 'weekday', 'weekend'

        reg_ids = session_used.query(AORegIds).filter_by(month=dep_date_month, tod=dep_date_tod, weekday_ind=dep_date_weekday).all()[0]  # TODO: WHY IS IT HERE [0]
        reg_ids = reg_ids.reg_id  # TODO: THIS BETTER HERE

        assert len(reg_ids) == 0, 'More reg_ids obtained than there should be'  # TODO: FIX THIS
        reg_id = int(reg_ids[0])

        drift_vol_avgp_raw = session_used.query(AOParam).filter_by(carrier=carrier, orig=orig, dest=dest, reg_id=reg_id).all()[0]  # TODO: just the first one - why

        if drift_vol_avgp_raw is None:  # nothing in the list
            # check if there is anything close to this,
            # reg_ids_similar is a list of reg_ids which are possible
            month_ranks, tod_ranks, weekday_ranks = find_close_regid( dep_date_month, dep_date_tod, dep_date_weekday)

            # month_ranks_str   = ','.join([ str(x)
            #                                for x in month_ranks ])  # making strings out of these lists
            # tod_ranks_str     = ','.join([ "'" + x + "'"
            #                                for x in tod_ranks ])
            # weekday_ranks_str = ','.join([ "'" + x + "'"
            #                                for x in weekday_ranks ])

            # mysql_conn_c.execute("""DELETE FROM reg_ids_temp;
            #                         INSERT INTO reg_ids_temp
            #                         SELECT reg_id
            #                         FROM reg_ids
            #                         WHERE month IN ({0}) AND weekday_ind = '{1}'
            #                         ORDER BY ABS(month - {2}) ASC, FIELD(tod, {3}) ASC, FIELD(weekday_ind, {4}) ASC;
            #
            #                         SET @reg_id_ord = (SELECT GROUP_CONCAT(reg_id) FROM reg_ids_temp);""".format( month_ranks_str
            #                                                                                                     , dep_date_weekday
            #                                                                                                     , dep_date_month
            #                                                                                                     , tod_ranks_str
            #                                                                                                     , weekday_ranks_str)
            #                     , multi = True)

            # FINISH THIS HERE!!!
            close_regs = session_used.query(AORegIds).filter( AORegIds.month.in_(month_ranks)
                                                            , AORegIds.weekday_ind.in_(weekday_ranks)
                                                            , AORegIds.month)
            # ordering going on

            # drift_vol_close_str = """
            # SELECT drift, vol, avg_price
            # FROM params
            # WHERE orig = '{0}' AND dest = '{1}' AND carrier = '{2}' AND
            #       reg_id IN (SELECT reg_id FROM reg_ids_temp)
            # ORDER BY FIELD(reg_id, @reg_id_ord)"""

            mysql_conn_c.execute(drift_vol_close_str.format( orig
                                                           , dest
                                                           , carrier))
            closest_drift_vol_res = mysql_conn_c.fetchall()

            if not closest_drift_vol_res:  # empty
                drift_vol_l.append(default_drift_vol)

            else:  # correct the drift
                drift_prelim, vol_prelim, avg_price = closest_drift_vol_res[0]  # take the first
                drift_vol_l.append(correct_drift_vol( drift_prelim
                                                  , vol_prelim
                                                  , default_drift_vol
                                                  , avg_price
                                                  , fwd_value))

        else:  # drift... has 1 entry
            drift_vol_l.append(correct_drift_vol( drift_vol_avgp_raw.drift
                                                , drift_vol_avgp_raw.vol
                                                , default_drift_vol
                                                , drift_vol_avgp_raw.avg_price
                                                , fwd_value))

    return drift_vol_l
