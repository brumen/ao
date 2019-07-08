# date manipulating functions
import datetime
import calendar as cal

from typing import List


def conv_local(date_str : str):
    """
    converts the string date into datetime for a single string

    :param date_str: date in the form of '20170501'
    :type date_str:  string
    :returns:     datetime.date version of the date
    :rtype:       datetime.date
    """
    return datetime.datetime(int(date_str[0:4]),
                             int(date_str[4:6]),
                             int(date_str[6:8]))


def convert_str_datetime(date_ : str) -> [datetime.date, List[datetime.date]]:
    """ Converts yyyymmdd into datetime.

    """

    if type(date_) is list:
        return [conv_local(d_elt) for d_elt in date_]

    # str. case
    return conv_local(date_)

    
def convert_str_dateslash(date_ : [str, List[str]]) -> [str, List[str]]:
    """ Converts yyyymmdd into date slash format

    :param date_: date in string 'yyyymmdd' format
    :returns:     date in / format '2017/05/05' or a list of them
    """

    def conv_local(d_elt):
        return str(int(d_elt[4:6])) + '/' + str(int(d_elt[6:8])) + '/' + str(int(d_elt[0:4]))

    if type(date_) is list:
        return [conv_local(d_elt) for d_elt in date_]

    return conv_local(date_)


def convert_str_date(date_ : [str, List[str]]) -> datetime.date:
    """ Converts yyyymmdd into datetime

    :param date_: date in yyyymmdd format
    :returns:     date in datetime.date format
    """

    def conv_local(d_elt):
        return datetime.date(int(d_elt[0:4]),
                             int(d_elt[4:6]),
                             int(d_elt[6:8]))
    
    if type(date_) is list:
        return [conv_local(d_elt) for d_elt in date_]

    return conv_local(date_)


def d2s(i : int) -> str:
    """ Digit to string conversion, adding 0 if the digit i < 10

    :param i: digit to be converted
    :returns: digit w/ possibly 0 prepended
    """

    return '0' + str(i) if i < 10 else str(i)


def convert_datetime_str(date_ : datetime.date) -> str:
    """ Converts the date in datetime format into string format

    :param date_: date in datetime.date format
    :returns:     same date in string format '20170502'
    """

    return str(date_.year) + d2s(date_.month) + d2s(date_.day)


def convert_dt_minus(date_ : datetime.date) -> str:
    """
    converts datetime.date to string date with - format

    :param date_: datetime.date in datetime.date format
    :returns:     date in - format '2017-05-02'
    """

    return str(date_.year) + '-' + d2s(date_.month) + '-' + d2s(date_.day)


def convert_dateslash_str(dates : str) -> str:
    """ Converts date in form 10/5/2016 -> 20161005

    """

    mon, day, year = dates.split('/')
    return year + d2s(int(mon)) + d2s(int(day))    


def convert_datedash_str(dates):
    """ Converts date in form 10-5-2016 -> 20161005.

    """

    mon, day, year = dates.split('-')

    return year + d2s(int(mon)) + d2s(int(day))


def convert_datedash_date(dates):
    """
    converts date in form 2016-10-5 -> dt.date(..)
    """

    year, mon, day = dates.split('-')
    return datetime.date(int(year), int(mon), int(day))


def convert_datedash_time_dt(date_i, hour_i):
    """

    returns:
    """

    year, mon, day = date_i.split('-')
    hour, minutes, sec = hour_i.split(':')
    return datetime.datetime( int(year)
                            , int(mon)
                            , int(day)
                            , int(hour)
                            , int(minutes))


def convert_date_datedash(_date : datetime.date) -> str:
    """
    Convert from datetime.date format to the same date in '2017-02-03' format.

    :param _date: date to be converted
    :returns:     date in datedash format
    """

    return '-'.join([ str(_date.year)
                    , d2s(_date.month)
                    , d2s(_date.day)])


def convert_hour_time(hour):
    """
    converts date in form 12:00:02 -> dt.time(..)

    """

    hour, minute, sec = hour.split(':')

    return datetime.time(int(hour), int(minute), int(sec))


def convert_dateslash_dash(dates : str) -> str:
    """
    converts date in form 10/5/2016 -> 2016-10-05

    :param dates: date in 10/5/2016 format
    :returns:     same date in the 2016-10-05 format
    """

    mon, day, year = dates.split('/')

    return '-'.join([year, d2s(int(mon)), d2s(int(day))])


def process_mm( m     : cal.monthcalendar
              , year  : int
              , month : int
              , day_b : int
              , day_e : int ) -> List[datetime.date] :
    """ Process the month matrix between day_b and day_e,
    A helper function for construct_date_range below.

    """

    T_l = []

    for row in m:
        for day in row:
            if (day >= day_b) and (day <= day_e):
                T_l.append(datetime.date(year, month, day))

    return T_l


def construct_date_range( date_b : datetime.date
                        , date_e : datetime.date ) -> List[datetime.date]:
    """ Constructs the date range between date_b and date_e, i.e.
    every date between these two dates.

    :param date_b: begin date (as in 20180317)
    :param date_e: end date (same as date_b
    :returns:      list of dates between date_b and date_e
    """

    year_b, month_b, day_b = date_b.year, date_b.month, date_b.day
    year_e, month_e, day_e = date_e.year, date_e.month, date_e.day

    T_l = []  # construction of the date list
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


def time_diff( date1 : [str, datetime.date]
             , date2 : [str, datetime.date]
             , dcf = 365.25 ) -> float:
    """
    computes numerical difference between date1 and date2 (date1 < date2)

    :param date1: first date (lower of the two dates), str in (20180105) format
    :param date2: second date (higher of the two), str in (20180509) format
    :returns: difference in numerical form
    """

    if isinstance(date1, datetime.datetime):
        return (date2 - date1).days / dcf

    return (convert_str_datetime(date2) - convert_str_datetime(date1)).days / dcf


def add_days_str(date_, days):
    """ Adds the number of days to date_

    """

    return convert_datetime_str(convert_str_datetime(date_) + datetime.timedelta(days=days))
