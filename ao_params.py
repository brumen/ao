# getting the params from the database
import config
import numpy as np
# air options modules 
import ds
import ao_codes
import ao_db
import mysql.connector 


def get_drift_vol_from_db(dep_date,  # in form '2017-02-15'
                          orig='SFO', dest='EWR', carrier='UA',
                          default_drift_vol=(500., 501.),
                          c_ao=ao_db.c_ao, conn_ao=ao_db.conn_ao,
                          correct_drift=False,
                          fwd_value=None):
    """
    pulls the drift and vol from database for the selected flight
    :param correct_drift: correct the drift, if negative make it positive 500, or so. 
    :param fwd_value: forward value used in case correct_drift == True
    """
    selected_drift_vol = """
    SELECT dep_date, drift, vol, avg_price FROM params WHERE orig= '%s' AND dest = '%s' AND carrier='%s' 
    """ % (orig, dest, carrier)
    res = ao_db.run_db(selected_drift_vol)
    if len(res) == 0:  # nothing in the list
        return default_drift_vol
    else:  # at least one entry, check if there are many, select most important
        # entry in form ('date time', drift, vol, avg_price)
        res_date, drift_prelim, vol_prelim, avg_price = select_closest_date(dep_date, res)  # return in the same format 
        # final correction in case desired
        return correct_drift_vol(drift_prelim, vol_prelim, default_drift_vol, correct_drift,
                                 fwd_value, avg_price)


def select_closest_date(date_desired, date_l):
    """
    selects the closest date to date_desired in the list date_l 
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


def correct_drift_vol(drift_prelim, vol_prelim, default_drift_vol, correct_drift,
                      fwd_price, avg_price):
    # corrects the drift and vol in case of nonsensical results 
    if (drift_prelim, vol_prelim) == (0., 0.):  # this means no change is in the database
        return default_drift_vol
    else:  # first drift, then vol
        if drift_prelim < 0.:
            drift_prelim = default_drift_vol[0]  # take a large default value, always change drift
        if not correct_drift:
            return drift_prelim, vol_prelim
        else:  # correct the drift/vol by rescaling it 
            scale_f = np.double(fwd_price)/avg_price
            return scale_f * drift_prelim, scale_f * vol_prelim


def get_drift_vol_from_db_precise(dep_date,  
                                  flight_dep_l, 
                                  orig='SFO', dest='EWR', carrier='UA',
                                  default_drift_vol=(500., 501.),
                                  c_ao=ao_db.c_ao, conn_ao=ao_db.conn_ao,
                                  correct_drift=True,
                                  fwd_value=None):
    """
    pulls the drift and vol from database for the selected flight, with more information than before
    :param dep_date: flight departure date, in the form of '2017-02-15' NOT SURE IF THIS NEEDED
    :param flight_dep_l: list of flights dep. dates & times in the form 2017-04-06T06:00:00
             they all share the same date, namely dep_date; so this is not important 
    :param correct_drift: correct the drift, if negative make it positive 500, or so. 
    :param fwd_value: forward value used in case correct_drift == True
    """
    flight_dep_time_l = [x.split('T') for x in flight_dep_l]  # list of dep. dates/times in form ('2017-06-06', '06:00:00')
    flight_dep_tod_l = [ao_codes.get_tod(x[1]) for x in flight_dep_time_l]  # list of 'morning' for all flights desired
    flight_dep_month_l = [ds.convert_datedash_date(d_t[0]).month for d_t in flight_dep_time_l]  # months extracted 
    flight_dep_weekday_l = [ao_codes.get_weekday_ind(ds.convert_datedash_date(d_t[0]).day)
                            for d_t in flight_dep_time_l]
    # month == integer
    # tod == text 'night'
    # weekday_ind == 'weekday', 'weekend'
    new_mysql = mysql.connector.connect(host='localhost', database='ao',
                                        user='brumen', password=ao_codes.brumen_mysql_pass)
    mysql_conn_c = new_mysql.cursor()

    reg_id_l = []
    drift_vol_l = []
    for dep_date_month, dep_date_tod, dep_date_weekday in zip(flight_dep_month_l,
                                                              flight_dep_tod_l,
                                                              flight_dep_weekday_l):
        first_attempt_reg_id = """SELECT reg_id FROM reg_ids
                                  WHERE month = %s AND tod = '%s' AND weekday_ind = '%s'""" \
                                    %(dep_date_month, dep_date_tod, dep_date_weekday)
        mysql_conn_c.execute(first_attempt_reg_id)

        # reg_id_l.append(int(mysql_conn_c.fetchone()[0]))
        reg_id = int(mysql_conn_c.fetchone()[0])  # there should be only one, yeah
        # first the general query, then the specific one 
        selected_drift_vol = """
        SELECT drift, vol, avg_price 
        FROM params 
        WHERE carrier = '%s' AND orig = '%s' AND dest = '%s' AND reg_id = %s
        """ % (carrier, orig, dest, reg_id)
        mysql_conn_c.execute(selected_drift_vol)
        drift_vol_avgp_raw = mysql_conn_c.fetchone()  # there is at most 1
        if drift_vol_avgp_raw is None:  # nothing in the list 
            # check if there is anything close to this,
            # reg_ids_similar is a list of reg_ids which are possible 
            month_ranks, tod_ranks, weekday_ranks = find_close_regid(dep_date_month, dep_date_tod, dep_date_weekday)  
            month_ranks_str = ','.join([str(x) for x in month_ranks])  # making strings out of these lists 
            tod_ranks_str = ','.join(["'" + x + "'" for x in tod_ranks])
            weekday_ranks_str = ','.join(["'" + x + "'" for x in weekday_ranks])
            find_reg_ids_str = """
                DELETE FROM reg_ids_temp; 

                INSERT INTO reg_ids_temp
                   SELECT reg_id 
                   FROM reg_ids 
                   WHERE month IN (%s) AND weekday_ind = '%s'
                   ORDER BY ABS(month - %s) ASC, FIELD(tod, %s) ASC, FIELD(weekday_ind, %s) ASC;
            
                SET @reg_id_ord = (SELECT GROUP_CONCAT(reg_id) FROM reg_ids_temp);""" \
                % (month_ranks_str, dep_date_weekday, 
                   dep_date_month,
                   tod_ranks_str,
                   weekday_ranks_str)
            mysql_conn_c.execute(find_reg_ids_str, multi=True)
            drift_vol_close_str = """
                SELECT drift, vol, avg_price 
                FROM params 
                WHERE orig = '%s' AND dest = '%s' AND carrier = '%s' AND 
                      reg_id IN (SELECT reg_id FROM reg_ids_temp)
                ORDER BY FIELD(reg_id, @reg_id_ord)""" \
                % (orig, dest, carrier)
            mysql_conn_c.execute(drift_vol_close_str)
            closest_drift_vol_res = mysql_conn_c.fetchall()
            if closest_drift_vol_res == []:  # empty
                drift_vol_l.append(default_drift_vol)  #
            else:  # correct
                drift_prelim, vol_prelim, avg_price = closest_drift_vol_res[0]  # take the first
                drift_vol_corr = correct_drift_vol(drift_prelim, vol_prelim, default_drift_vol,
                                                   correct_drift, fwd_value, avg_price)
                drift_vol_l.append(drift_vol_corr)
        else:  # drift... has 1 entry 
            drift_prelim, vol_prelim, avg_price = drift_vol_avgp_raw
            drift_vol_corr = correct_drift_vol(drift_prelim, vol_prelim, default_drift_vol,
                                              correct_drift, fwd_value, avg_price)
            drift_vol_l.append(drift_vol_corr)
            
    return drift_vol_l


def find_close_regid(month, tod, weekday):
    """
    logic how to select regid which is close to desired one
    """
    # months allowed +/- 2
    def rot_month(m, k):
        # rotational month
        return ((m - 1) + k) % 12 + 1
    # months considered 
    month_ranks = [month, rot_month(month, 1), rot_month(month, -1), rot_month(month, 2), rot_month(month, -2)]
    # tod ranking
    all_ranks = ['morning', 'afternoon', 'evening', 'night']
    tod_idx = all_ranks.index(tod)
    tod_ranks = [all_ranks[tod_idx], all_ranks[(tod_idx+1)%4],
                 all_ranks[(tod_idx+2)%4], all_ranks[(tod_idx+3)%4]]
    if weekday == 'weekday':  # ranking of weekdays
        weekday_ranks = ['weekday', 'weekend']
    else:
        weekday_ranks = ['weekend', 'weekday']

    return month_ranks, tod_ranks, weekday_ranks

