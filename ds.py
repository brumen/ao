# date manipulating functions
import datetime
import calendar as cal

from typing import List


def d2s(i : int) -> str:
    """ Digit to string conversion, adding 0 if the digit i < 10

    :param i: digit to be converted
    :returns: digit w/ possibly 0 prepended
    """

    return '0' + str(i) if i < 10 else str(i)


def convert_datedash_date(date_ : str) -> datetime.date:
    """ converts date in form 2016-10-5 -> dt.date(..)

    :param date_: date in the format described.
    :returns: the same date in datetime.date format
    """

    year, mon, day = date_.split('-')

    return datetime.date(int(year), int(mon), int(day))


def convert_datedash_time_dt(date_i, hour_i):
    """

    returns:
    """

    year, mon, day     = date_i.split('-')
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

    return '-'.join([ str(_date.year), d2s(_date.month), d2s(_date.day)])


def convert_hour_time(hour):
    """
    converts date in form 12:00:02 -> dt.time(..)

    """

    hour, minute, sec = hour.split(':')

    return datetime.time(int(hour), int(minute), int(sec))


def process_mm( m     : cal.monthcalendar
              , year  : int
              , month : int
              , day_b : int
              , day_e : int ) -> List[datetime.date]:
    """ Process the month matrix between day_b and day_e,
    A helper function for construct_date_range below.

    :param m: calendar month
    :param year: year TODO: FINISH HERE
    :param month:
    :param day_b: begin day
    :param day_e: end day
    """

    date_list = []
    for row in m:
        for day in row:
            if (day >= day_b) and (day <= day_e):
                date_list.append(datetime.date(year, month, day))

    return date_list


def construct_date_range( date_b : datetime.date
                        , date_e : datetime.date ) -> List[datetime.date]:
    """ Constructs the date range between date_b and date_e, i.e.
    every date between these two dates.

    :param date_b: begin date (as in 20180317)
    :param date_e: end date (same as date_b
    :returns: list of dates between date_b and date_e
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
