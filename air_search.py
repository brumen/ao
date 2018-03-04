# air option search and compute 

import time
import datetime            as dt

from requests              import ConnectionError
from skyscanner.skyscanner import Flights
from skyscanner.skyscanner import FlightsCache

# AirOption files
import ds
import ao_codes
from   ao_codes            import COUNTRY, CURRENCY, LOCALE
from   mysql_connector_env import MysqlConnectorEnv


def get_itins( origin_place    = 'SIN'
             , dest_place      = 'KUL'
             , outbound_date   = None
             , includecarriers = None
             , cabinclass      = 'Economy'
             , adults          = 1
             , use_cache       = False
             , nb_tries        = 1 ):
    """
    helper function that returns itineraries, uses skyscanner api 

    :param origin_place:  IATA code of the flight origin airport
    :type origin_place:   str
    :param dest_place:    IATA code of the flight destination airport
    :type dest_place:     str
    :param outbound_date: date for flights to fetch
    :type outbound_date:  datetime.date
    :param includecarriers: IATA code of the airline to use
    :param cabinclass:    one of the following: Economy*, PremiumEconomy, Business, First
    :type cabinclass:     str
    :param nb_tries:      number of tries that one tries to get a connection to SkyScanner
    :type nb_tries:       int
    :returns:          TODO
    :rtype:            TODO
    """

    params_all = dict( country          = COUNTRY
                     , currency         = CURRENCY
                     , locale           = LOCALE
                     , originplace      = origin_place + '-sky'
                     , destinationplace = dest_place + '-sky'
                     , outbounddate     = ds.convert_date_datedash(outbound_date)
                     , cabinclass       = cabinclass
                     , adults           = adults )

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
        params_all['market'] = COUNTRY  # add this field
        
    try:
        result = query_fct(**params_all).parsed
    except (ConnectionError, AttributeError):
        time.sleep(5)  # wait 5 secs
        if nb_tries <= 5:
            return get_itins( origin_place    = origin_place
                            , dest_place      = dest_place
                            , outbound_date   = outbound_date
                            , includecarriers = includecarriers
                            , cabinclass      = cabinclass
                            , adults          = adults
                            , nb_tries        = nb_tries + 1 )
        else:
            return None  # this is handled appropriately in the get_ticket_prices

    return result


def find_carrier(carrier_l, carrier_id):
    """
    Finds the carrier from the ID list

    :param carrier_l:  list of carriers
    :type carrier_l:   list of str
    :param carrier_id: carrier one is searching for
    :type carrier_id:  str
    :returns:          Code of the carrier info
    :rtype:            None if failure; carrier_info if carrier is found
    """

    for carrier_info in carrier_l:
        if carrier_id == carrier_info['Id']:
            return carrier_info['Code']

    return None  # None indicates failure


def get_cached_flights(flights_in_ldb):
    """
    Obtains the flight data from the local database

    :param flights_in_ldb: flights as obtained from a local database.
    :type flights_in_ldb: TODO:
    :returns: F_v, flights_v_str, reorg_flights_v
    :rtype: TODO
    """

    # construct F_v, flights_v, reorg_flights_v
    F_v = [x[5] for x in flights_in_ldb]
    flights_v_str = []

    for fl in flights_in_ldb:
        # dep_date/hour
        dep_date_str = fl[2].isoformat()
        dep_time_str = (dt.datetime.min + fl[3]).time().isoformat()  # converts to string
        flights_v_str.append((fl[1],
                              dep_date_str + 'T' + dep_time_str,
                              fl[4].isoformat(), fl[5], fl[6] + fl[7]))

    return F_v, flights_v_str, reorganize_ticket_prices(flights_v_str)


def insert_into_flights_live( origin_place
                            , dest_place
                            , flights_v
                            , cabinclass ):
    """
    Inserts the flights given in flights_v into the flights_live database

    :param origin_place: IATA code of the origin airport
    :type origin_place: str
    :param dest_place: IATA code of the destination airport
    :type dest_place: TODO
    """

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
        live_fl_l.append(
            (as_of, origin_place, dest_place, it[3], it[0], dep_date, dep_time, it[2], carrier, flight_nb, cabinclass))

    insert_str = """INSERT INTO flights_live (as_of, orig, dest, price, flight_id, dep_date, dep_time, 
                                              arr_date, carrier, flight_nb, cabin_class) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""

    with MysqlConnectorEnv() as mysql_conn:
        mysql_conn.cursor().executemany(insert_str, live_fl_l)
        mysql_conn.commit()


def extract_Fv_flights_from_results(result):
    """
    Extracts the flight forward prices and flight data from the results provided

    :param results:
    :type results:
    :returns:
    :rtype: tuple ( TODO  )
    """
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

    return F_v, flights_v


def get_ticket_prices( origin_place
                     , dest_place
                     , outbound_date
                     , include_carriers   = None
                     , cabinclass         = 'Economy'
                     , adults             = 1
                     , use_cache          = False
                     , insert_into_livedb = False ):
    """
    Returns the ticket prices for a flight

    :param origin_place: IATA code of the origin airport 'SIN'
    :type origin_place: str
    :param dest_place: IATA code of the destination airport 'KUL'
    :type dest_place: str
    :param outbound_date: outbound date # TODO: remove: in dash format '2017-02-15'
    :type outbound_date: datetime.date
    :param include_carriers: IATA code of a _SINGLE_ airline code
    :type include_carriers: str
    :param cabinclass: cabin class of the flight ticket (one of 'Economy', 'Business')
    :type cabinclass: str
    :param adults: number of adult tickets booked
    :type adults: int
    :param: use_cache:
    :type use_cache: bool
    :param insert_into_livedb: indicator whether to insert the fetched flight into the livedb
    :type insert_into_livedb:  bool
    :returns: TODO
    :rtype:
    """

    # local time
    lt = time.localtime()
    lt_dt = dt.datetime(lt.tm_year, lt.tm_mon, lt.tm_mday, lt.tm_hour, lt.tm_min, lt.tm_sec)
    lt_adj = (lt_dt - ao_codes.livedb_delay).isoformat()
    
    # first check the local database
    flights_live_local = """
    SELECT as_of, flight_id, dep_date, dep_time, arr_date, price, carrier, flight_nb 
    FROM flights_live 
    WHERE orig = '{0}' AND dest = '{1}' AND dep_date = '{2}' AND cabin_class = '{3}' AND as_of > '{4}' 
    """.format(origin_place, dest_place, outbound_date, cabinclass, lt_adj)

    if include_carriers is not None:
        flights_live_local += "  AND carrier = '{0}'".format(include_carriers)  # include_carriers is only 1 in this case

    with MysqlConnectorEnv() as mysql_conn:
        mysql_curr = mysql_conn.cursor()
        mysql_curr.execute(flights_live_local)
        flights_in_ldb = mysql_curr.fetchall()

    if flights_in_ldb:  # we have this in the database
        return get_cached_flights(flights_in_ldb)

    # If flights not found in the cached db, continue with skyscanner fetch
    result = get_itins( origin_place    = origin_place
                      , dest_place      = dest_place
                      , outbound_date   = outbound_date
                      , includecarriers = include_carriers
                      , cabinclass      = cabinclass
                      , adults          = adults
                      , use_cache       = use_cache)

    # returns all one ways on that date
    if result is None:  # nothing out
        return [], [], []
    else:  # results are not empty
        F_v, flights_v = extract_Fv_flights_from_results(result)

        if insert_into_livedb:  # insert obtained flights into livedb
            insert_into_flights_live(origin_place, dest_place, flights_v, cabinclass)

        return F_v, flights_v, reorganize_ticket_prices(flights_v)


def reorganize_ticket_prices(itin):
    """
    reorganize ticket prices by levels:
       day
         time of day (morning, afternoon)
            hour 

        insert ((date, hour), (arr_date, arr_hour), price, flight_id) into dict d

    :param itin: Itinerary in the form of a list of ((u'2016-10-28', u'19:15:00'), 532.),
                   where the first is the departure date, second departure time, third flight price
    :type itin: list of (tuple, double)
    :returns: dictionary as of the form as described above in the function description
    :rtype: dict
    """

    # get the days from the list of
    dep_day_hour = [(x[1].split('T'), x[2].split('T'), x[3], x[0], x[4]) for x in itin]

    reorgTickets = dict()

    for (date, hour), (arr_date, arr_hour), price, flight_id, flight_num in dep_day_hour:
        time_of_day_res = ao_codes.get_tod(hour)

        date_dt = date  # TODO: CHECK IF THIS HERE IS CORRECT ds.convert_datedash_date(date)

        # part to insert into dict d
        if date_dt not in reorgTickets.keys():
            reorgTickets[date_dt] = dict()
        if time_of_day_res not in reorgTickets[date_dt].keys():
            reorgTickets[date_dt][time_of_day_res] = dict()
        # TODO: True is to follow the flights in the app
        reorgTickets[date_dt][time_of_day_res][hour] = (flight_id, date_dt, hour, arr_date, arr_hour, price, flight_num, True)

    return reorgTickets


def get_all_carriers( origin_place
                    , dest_place
                    , outbound_date
                    , cabinclass    = 'Economy'):
    """
    gets all carriers for a selected route and selected date (direct flights only)

    :param origin_place: IATA code of the origin airport
    :type origin_place:  str
    :param dest_place:   IATA code of the destination airport
    :type dest_place:    str
    :param outbound_date: date of the flights between origin, destination
    :type outbound_date:  str (in '2017-02-05') format

    """

    all_data = get_ticket_prices( origin_place  = origin_place
                                , dest_place    = dest_place
                                , outbound_date = outbound_date
                                , cabinclass    = cabinclass)

    carrier_set = set([flight[4][:2]
                       for flight in all_data[1]])  # carrier names, a set

    return carrier_set
