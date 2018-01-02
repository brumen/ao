# date manipulating functions
import numpy as np
import scipy.interpolate
import datetime as dt
import calendar as cal


def conv_local(date_str):
    """
    converts the string date into datetime for a single string

    :param date_str: date in the form of '20170501'
    :type date_str:  string
    :returns:     datetime.date version of the date
    :rtype:       datetime.date
    """
    return dt.datetime(int(date_str[0:4]),
                       int(date_str[4:6]),
                       int(date_str[6:8]))


def convert_str_datetime(date_):
    """
    converts yyyymmdd into datetime
    """

    if type(date_) is list:
        return [conv_local(d_elt) for d_elt in date_]
    else:
        return conv_local(date_)

    
def convert_str_dateslash(date_):
    """
    converts yyyymmdd into date slash format
    """
    def conv_local(d_elt):
        return str(int(d_elt[4:6])) + '/' + str(int(d_elt[6:8])) + '/' + str(int(d_elt[0:4]))

    if type(date_) is list:
        return [conv_local(d_elt) for d_elt in date_]
    else:
        return conv_local(date_)

    

def convert_str_date(date_):
    """
    converts yyyymmdd into datetime
    """
    def conv_local(d_elt):
        return dt.date(int(d_elt[0:4]),
                       int(d_elt[4:6]),
                       int(d_elt[6:8]))
    
    if type(date_) is list:
        return [conv_local(d_elt) for d_elt in date_]
    else:
        return conv_local(date_)


def d2s(i):
    """
    digit to string conversion, adding 0 if < 10
    """
    if i < 10:
        return "0" + str(i)
    else:
        return str(i)


def convert_datetime_str(date_):
    """
    converts the date in datetime format into string format 
    """
    return str(date_.year) + d2s(date_.month) + d2s(date_.day)


def convert_dt_minus(date_):
    return str(date_.year) + '-' + d2s(date_.month) + '-' + d2s(date_.day)


def convert_dateslash_str(dates):
    """
    converts date in form 10/5/2016 -> 20161005
    """
    mon, day, year = dates.split('/')
    return year + d2s(int(mon)) + d2s(int(day))    


def convert_datedash_str(dates):
    """
    converts date in form 10-5-2016 -> 20161005
    """
    mon, day, year = dates.split('-')
    return year + d2s(int(mon)) + d2s(int(day))    


def convert_datedash_date(dates):
    """
    converts date in form 2016-10-5 -> dt.date(..)
    """
    year, mon, day = dates.split('-')
    return dt.date(int(year), int(mon), int(day))


def convert_datedash_time_dt(date_i, hour_i):
    """
    returns 
    """
    year, mon, day = date_i.split('-')
    hour, minutes, sec = hour_i.split(':')
    return dt.datetime(int(year), int(mon), int(day), int(hour), int(minutes))


def convert_hour_time(hour):
    """
    converts date in form 12:00:02 -> dt.time(..)
    """
    hour, minute, sec = hour.split(':')
    return dt.time(int(hour), int(minute), int(sec))


def convert_dateslash_dash(dates):
    """
    converts date in form 10/5/2016 -> 2016-10-05
    """
    mon, day, year = dates.split('/')
    return year + '-' + d2s(int(mon)) + '-' + d2s(int(day))    


def construct_date_range(date_b, date_e):
    """
    constructs the date range between date_b and date_e
    :param date_b: begin date, in string format
    :param date_e: end date, in string format
    """
    date_b_dt = convert_str_date(date_b)
    date_e_dt = convert_str_date(date_e)
    year_b, month_b, day_b = date_b_dt.year, date_b_dt.month, date_b_dt.day
    year_e, month_e, day_e = date_e_dt.year, date_e_dt.month, date_e_dt.day
    T_l = []  # construction of the date list 

    def process_mm(m, year, month, day_b, day_e):
        """
        process the month matrix between day_b and day_e
        """
        T_l = []
        for row in m:
            for day in row:
                if day >= day_b and day <= day_e:
                    T_l.append(dt.date(year, month, day))
        return T_l
        
    for year in range(year_b, year_e+1):
        if year == year_b:
            if year_e == year_b:
                month_ends = month_e
            else:
                month_ends = 12
            # month_ends assumes the role of month_e
            for month in range(month_b, month_ends+1):
                mm = cal.monthcalendar(year, month)  # month matrix
                if month == month_b:  # beginning year, beginning month 
                    if month == month_ends:  # beginning and end month are the same
                        T_l.extend(process_mm(mm, year, month, day_b, day_e))
                    else:
                        T_l.extend(process_mm(mm, year, month, day_b, 31))
                elif month == month_e:
                    T_l.extend(process_mm(mm, year, month, 1, day_e))
                else:
                    T_l.extend(process_mm(mm, year, month, 1, 31))
        elif year == year_e:
            for month in range(1, month_e+1):
                mm = cal.monthcalendar(year, month)  # month matrix
                if month == month_e:
                    T_l.extend(process_mm(mm, year, month, 1, day_e))
                else:
                    T_l.extend(process_mm(mm, year, month, 1, 31))
        else:
            for month in range(1, 13):  # all months 
                mm = cal.monthcalendar(year, month)  # month matrix
                T_l.extend(process_mm(mm, year, month, 1, 31))

    return T_l


def time_diff(date1, date2, dt_format=365.25):
    """
    computes numerical difference between date1 date2
    """
    if type(date1) is dt.datetime:
        return (date2 - date1).days / dt_format
    else:
        date1_dt = convert_str_datetime(date1)
        date2_dt = convert_str_datetime(date2)
        return (date2_dt - date1_dt).days / dt_format


def add_days_str(date_, days):
    """
    adds the number of days to date_
    """
    return convert_datetime_str(convert_str_datetime(date_) + dt.timedelta(days=days))
