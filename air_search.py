# air option search and compute 

import time
from requests              import ConnectionError
import mysql.connector
import datetime            as dt

from skyscanner.skyscanner import Flights
from skyscanner.skyscanner import FlightsCache

# AirOption files
import ds
import ao_codes

# common variables
MYSQL_HOST = 'localhost'
MYSQL_DB   = 'ao'
MYSQL_USER = 'brumen'


def get_itins(origin_place    = 'SIN',
              dest_place      = 'KUL',
              outbound_date   = '2017-02-05',
              country         = 'US',
              currency        = 'USD',
              locale          = 'en-US',
              includecarriers = None,
              cabinclass      = 'Economy',
              adults          = 1,
              use_cache       = False,
              nb_tries        = 1):
    """
    helper function that returns itineraries, uses skyscanner api 

    :param origin_place: origin of the flight
    :type origin_place:  string
    :param dest_place:   destination of the flight
    :type dest_place:    string
    :param cabinclass: one of the following: Economy*, PremiumEconomy, Business, First
    :type cabinclass:  String
    :param nb_tries:   number of tries that one tries to get a connection to SkyScanner
    :type nb_tries:    integer
    """

    origin_place_used = origin_place + '-sky'
    dest_place_used = dest_place + '-sky'
    
    params_all = dict(country          = country,
                      currency         = currency,
                      locale           = locale,
                      originplace      = origin_place_used,
                      destinationplace = dest_place_used,
                      outbounddate     = outbound_date,
                      cabinclass       = cabinclass,
                      adults           = adults)

    if includecarriers is not None:
        params_all['includecarriers'] = includecarriers
    
    if not use_cache:
        flights_service = Flights(ao_codes.skyscanner_api_key)
        query_fct = flights_service.get_result
    else:
        flights_service = FlightsCache(ao_codes.skyscanner_api_key)
        query_fct = flights_service.get_cheapest_price_by_route
        # query_fct = flights_service.get_cheapest_quotes
        # query_fct = flights_service.get_cheapest_price_by_date
        # query_fct = flights_service.get_grid_prices_by_date
        params_all['market'] = country  # add this field
        
    try:
        result = query_fct(**params_all).parsed
    except (ConnectionError, AttributeError):
        time.sleep(5)  # wait 5 secs
        # result = query_fct(**params_all).parsed
        if nb_tries <= 5:
            return get_itins( origin_place    = origin_place
                            , dest_place      = dest_place
                            , outbound_date   = outbound_date
                            , country         = country
                            , currency        = currency
                            , locale          = locale
                            , includecarriers = includecarriers
                            , cabinclass      = cabinclass
                            , adults          = adults
                            , use_cache       = use_cache
                            , nb_tries        = nb_tries+1)
        else:
            return None  # this is handled appropriately in the get_ticket_prices

    return result


def find_carrier(carrier_l, carrier_id):
    """
    finds the carrier from the ID list

    :param carrier_l:  list of carriers
    :type carrier_l:   list of strings
    :param carrier_id: carrier one is searching for
    :type carrier_id:  string
    :returns:          Code of the carrier info
    :rtype:            TODO: ???
    """

    for carrier_info in carrier_l:
        if carrier_id == carrier_info['Id']:
            return carrier_info['Code']
    return None  # this should never be reached


def get_ticket_prices(origin_place       = 'SIN',
                      dest_place         = 'KUL',  # possible NYCA-sky
                      outbound_date      = '2017-03-05',
                      country            = 'US',
                      currency           = 'USD',
                      locale             = 'en-US',
                      include_carriers   = None,  # possible 'SQ' - singapore airlines
                      cabinclass         = 'Economy',
                      adults             = 1,
                      use_cache          = False,
                      insert_into_livedb = False,
                      use_mysql_conn     = None):
    """
    returns the ticket prices for a flight 

    """

    # local time
    lt = time.localtime()
    lt_dt = dt.datetime(lt.tm_year, lt.tm_mon, lt.tm_mday, lt.tm_hour, lt.tm_min, lt.tm_sec)
    lt_adj = (lt_dt - ao_codes.livedb_delay).isoformat()
    
    # first check the local database
    flights_live_local = """
    SELECT as_of, flight_id, dep_date, dep_time, arr_date, price, carrier, flight_nb FROM flights_live 
    WHERE orig = '{0}' AND dest = '{1}' AND dep_date = '{2}' AND cabin_class = '{3}' AND as_of > '{4}' 
    """.format(origin_place, dest_place, outbound_date, cabinclass, lt_adj)
    if include_carriers is not None:
        flights_live_local += "  AND carrier = '{0}'".foramt(include_carriers)  # include_carriers is only 1 in this case

    # connect to the mysql and retrieve this 
    if use_mysql_conn is None: 
        mysql_conn = mysql.connector.connect( host     = MYSQL_HOST
                                            , database = MYSQL_DB
                                            , user     = MYSQL_USER
                                            , password = ao_codes.brumen_mysql_pass)
    else:
        mysql_conn = use_mysql_conn

    mysql_c = mysql_conn.cursor()
    mysql_c.execute(flights_live_local)
    flights_in_ldb = mysql_c.fetchall()
    
    if len(flights_in_ldb) > 0:  # we have this in the database
        # construct F_v, flights_v, reorg_flights_v
        F_v = [x[5] for x in flights_in_ldb]
        flights_v_str = []
        for fl in flights_in_ldb:
            # dep_date/hour 
            dep_date_str = fl[2].isoformat()  # ds.convert_dt_minus(fl_ch[2].date())
            dep_time_str = (dt.datetime.min + fl[3]).time().isoformat()  # converts to string
            flights_v_str.append((fl[1],
                                  dep_date_str + 'T' + dep_time_str,
                                  fl[4].isoformat(), fl[5], fl[6] + fl[7]))

        reorg_flights_v = reorganize_ticket_prices(flights_v_str)
        return F_v, flights_v_str, reorg_flights_v

    # otherwise continue here with skyscanner search 
    result = get_itins(origin_place    = origin_place,
                       dest_place      = dest_place,
                       outbound_date   = outbound_date,
                       country         = country,
                       currency        = currency,
                       locale          = locale,
                       includecarriers = include_carriers,
                       cabinclass      = cabinclass,
                       adults          = adults,
                       use_cache       = use_cache)

    # returns all one ways on that date
    if result is None:  # nothing out
        return [], [], []
    else:
        ri = result['Itineraries']
        rl = result['Legs']  # legs and itineraries are the same in length
        carriers = result['Carriers']
        F_v = []
        flights_v = []
        for itin, leg in zip(ri, rl):
            # determine if the flight is direct
            direct_ind = len(leg['FlightNumbers']) == 1  # indicator if the flight is direct
            outbound_leg_id = leg['Id']
            dep_date = leg['Departure']
            arr_date = leg['Arrival']
            carrier_id_l = leg['Carriers']  # [0] (multiple carriers, list)
            po = itin['PricingOptions']
            flight_num_all = leg['FlightNumbers']
            if direct_ind:  # the other test case is missing 
                carrier_id = carrier_id_l[0]  # first (and only) carrier
                carrier = find_carrier(carriers, carrier_id)
                price = po[0]["Price"]  # TODO: THIS PRICE CAN BE DIFFERENT
                flight_num = flight_num_all[0]['FlightNumber']
                F_v.append(price)
                flights_v.append((outbound_leg_id, dep_date, arr_date, price, carrier + flight_num))

        reorg_flights_v = reorganize_ticket_prices(flights_v)

        if insert_into_livedb:  # insert obtained flights into livedb 
            # construct ins_fl_l
            lt = time.localtime()
            as_of = str(lt.tm_year) + '-' + str(ds.d2s(lt.tm_mon)) + '-' + str(ds.d2s(lt.tm_mday)) + 'T' + \
                    str(ds.d2s(lt.tm_hour)) + ':' + str(ds.d2s(lt.tm_min)) + ':' + str(ds.d2s(lt.tm_sec))
            live_fl_l = []
            for it in flights_v:
                dep_date, dep_time = it[1].split('T')
                # arr_date = it[2]
                carrier = it[4][:2]  # first 2 letters of this string
                flight_nb = it[4][3:]  # next letters for flight_nb
                live_fl_l.append((as_of, origin_place, dest_place, it[3], it[0], dep_date, dep_time, it[2], carrier, flight_nb, cabinclass))

            mysql_conn = mysql.connector.connect( host     = MYSQL_HOST
                                                , database = MYSQL_DB
                                                , user     = MYSQL_USER
                                                , password = ao_codes.brumen_mysql_pass)
            mysql_c = mysql_conn.cursor()
            insert_str = """INSERT INTO flights_live (as_of, orig, dest, price, flight_id, dep_date, dep_time, 
                            arr_date, carrier, flight_nb, cabin_class) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
            mysql_c.executemany(insert_str, live_fl_l)
            mysql_conn.commit()

        return F_v, flights_v, reorg_flights_v


def reorganize_ticket_prices(itin):
    """
    reorganize ticket prices by levels:
       day
         time of day (morning, afternoon)
            hour 

        insert ((date, hour), (arr_date, arr_hour), price, flight_id) into dict d
    """

    # get the days from the list of (u'2016-10-28', u'19:15:00'), 532.
    dep_day_hour = [(x[1].split('T'), x[2].split('T'), x[3], x[0], x[4]) for x in itin]
    reorgTickets = dict()

    for date_hour, arr_date_hour, price, flight_id, flight_num in dep_day_hour:
        date, hour = date_hour  # departure date
        arr_date, arr_hour = arr_date_hour
        time_of_day_res = ao_codes.get_tod(hour)

        # part to insert into dict d
        if date not in reorgTickets.keys():
            reorgTickets[date] = dict()
        if time_of_day_res not in reorgTickets[date].keys():
            reorgTickets[date][time_of_day_res] = dict()
        # TODO: True is to follow the flights in the app
        reorgTickets[date][time_of_day_res][hour] = (flight_id, date, hour, arr_date, arr_hour, price, flight_num, True)

    return reorgTickets


def get_all_carriers(origin_place='SIN',
                     dest_place='KUL',
                     outbound_date='2017-02-05',
                     country='US',
                     currency='USD',
                     locale='en-US',
                     cabinclass='Economy'):
    """
    gets all carriers for a selected route and selected date (direct flights only)

    """
    all_data = get_ticket_prices(origin_place  = origin_place,
                                 dest_place    = dest_place,
                                 outbound_date = outbound_date,
                                 country       = country,
                                 currency      = currency,
                                 locale        = locale,
                                 cabinclass    = cabinclass,
                                 errors        = 'graceful')

    carrier_set = set([flight[4][:2] for flight in all_data[1]])  # carrier names, a set
    return carrier_set
