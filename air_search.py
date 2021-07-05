# air option search and compute

import time
import datetime

from typing import Tuple, Union, List, Dict

from requests              import ConnectionError
from skyscanner.skyscanner import Flights
from skyscanner.skyscanner import FlightsCache

# AirOption files
from ao.ds                  import convert_date_datedash, d2s
from ao.ao_codes            import COUNTRY, CURRENCY, LOCALE, skyscanner_api_key, livedb_delay, get_tod
from ao.mysql_connector_env import MysqlConnectorEnv


def get_itins( origin_place    : str
             , dest_place      : str
             , outbound_date   : datetime.date
             , includecarriers : Union[List[str], None] = None
             , cabinclass      : str                    = 'Economy'
             , adults          : int                    = 1
             , use_cache       : bool                   = False
             , nb_tries        : int                    = 1
             , max_nb_tries    : int                    = 5 ) -> Union[Dict, None]:
    """ Returns itineraries for the selection from the Skyscanner API.

    :param origin_place:  IATA code of the flight origin airport (e.g. 'SIN', or 'SFO')
    :param dest_place:    IATA code of the flight destination airport (e.g. 'KUL', or 'EWR')
    :param outbound_date: date for flights to fetch
    :param includecarriers: IATA code of the airlines to use, if None, all airlines
    :param cabinclass:    one of the following: Economy*, PremiumEconomy, Business, First
    :param adults: number of adults to get
    :param use_cache: whether to use Skyscanner cache for ticket pricer. This is not the local db, just cache part of Skyscanner
    :param nb_tries:      number of tries that one tries to get a connection to SkyScanner
    :param max_nb_tries: max number of tries that it attempts.
    :returns:             Resulting flights from SkyScanner, dictionary structure:
                          'Itineraries'
                          'Currencies'
                          'Agents'
                          'Carriers'
                          'Query'
                          'Segments'
                          'Places'
                          'SessionKey'
                          'Legs'
                          'Status'
    """

    params_all = dict( country          = COUNTRY
                     , currency         = CURRENCY
                     , locale           = LOCALE
                     , originplace      = origin_place + '-sky'
                     , destinationplace = dest_place + '-sky'
                     , outbounddate     = convert_date_datedash(outbound_date)
                     , cabinclass       = cabinclass
                     , adults           = adults
                     , stops            = 0 )  # only direct flights

    if includecarriers is not None:
        params_all['includecarriers'] = includecarriers

    if not use_cache:
        flights_service = Flights(skyscanner_api_key)
        query_fct = flights_service.get_result
    else:
        flights_service = FlightsCache(skyscanner_api_key)
        query_fct = flights_service.get_cheapest_price_by_route
        # query_fct = flights_service.get_cheapest_quotes
        # query_fct = flights_service.get_cheapest_price_by_date
        # query_fct = flights_service.get_grid_prices_by_date
        params_all['market'] = COUNTRY  # add this field

    try:
        return query_fct(**params_all).parsed

    except (ConnectionError, AttributeError):
        time.sleep(5)  # wait 5 secs
        if nb_tries <= max_nb_tries:
            return get_itins( origin_place    = origin_place
                            , dest_place      = dest_place
                            , outbound_date   = outbound_date
                            , includecarriers = includecarriers
                            , cabinclass      = cabinclass
                            , adults          = adults
                            , nb_tries        = nb_tries + 1
                            , max_nb_tries    = max_nb_tries )

        return None  # this is handled appropriately in the get_ticket_prices


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


def get_cached_flights(flights_in_ldb) -> Tuple:
    """ Obtains the flight data from the local database

    :param flights_in_ldb: flights as obtained from a local database.
    :returns: F_v, flights_v_str
    """

    # construct F_v, flights_v, reorg_flights_v
    F_v = [x[5] for x in flights_in_ldb]
    flights_v_str = []

    for fl in flights_in_ldb:
        # dep_date/hour
        dep_date_str = fl[2].isoformat()
        dep_time_str = (datetime.datetime.min + fl[3]).time().isoformat()  # converts to string
        flights_v_str.append((fl[1],
                              dep_date_str + 'T' + dep_time_str,
                              fl[4].isoformat(), fl[5], fl[6] + fl[7]))

    return F_v, flights_v_str


def insert_into_flights_live( origin : str
                            , dest   : str
                            , flights
                            , cabinclass ):
    """
    Inserts the flights given in flights_v into the flights_live database

    :param origin: IATA code of the origin airport
    :param dest: IATA code of the destination airport
    :param flights: flights to be insterted
    :param cabinclass: cabin class ('economy',...)
    """

    # construct ins_fl_l
    lt = time.localtime()
    as_of = str(lt.tm_year) + '-' + str(d2s(lt.tm_mon)) + '-' + str(d2s(lt.tm_mday)) + 'T' + \
            str(d2s(lt.tm_hour)) + ':' + str(d2s(lt.tm_min)) + ':' + str(d2s(lt.tm_sec))

    live_flight_l = []

    for it in flights:

        dep_date, dep_time = it[1].split('T')
        # arr_date = it[2]
        carrier = it[4][:2]  # first 2 letters of this string
        flight_nb = it[4][3:]  # next letters for flight_nb
        live_flight_l.append(
            (as_of, origin, dest, it[3], it[0], dep_date, dep_time, it[2], carrier, flight_nb, cabinclass))

    insert_str = """INSERT INTO flights_live (as_of, orig, dest, price, flight_id, dep_date, dep_time, 
                                              arr_date, carrier, flight_nb, cabin_class) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""

    with MysqlConnectorEnv() as mysql_conn:
        mysql_conn.cursor().executemany(insert_str, live_flight_l)
        mysql_conn.commit()


def extract_Fv_flights_from_results(result) -> Tuple:
    """ Extracts the flight forward prices and flight data from the results provided

    :param result: result of output from SkyScanner, dictionary structure:
                          'Itineraries'
                          'Currencies'
                          'Agents'
                          'Carriers'
                          'Query'
                          'Segments'
                          'Places'
                          'SessionKey'
                          'Legs'
                          'Status'
    :type result: dict
    :returns:
    """

    F_v = []
    flights_v = []

    for itinerary, leg in zip(result['Itineraries'], result['Legs']):

        flight_num_all = leg['FlightNumbers']

        if len(flight_num_all) == 1:  # indicator if the flight is direct, the other test case is missing
            carrier = find_carrier(result['Carriers'], leg['Carriers'][0])  # carriers = all carriers, leg['carriers'] are id of carrier
            price = itinerary['PricingOptions'][0]['Price']  # TODO: THIS PRICE CAN BE DIFFERENT
            flight_num = flight_num_all[0]['FlightNumber']
            F_v.append(price)
            # leg['Departure'] is departure date
            flights_v.append((leg['Id'], leg['Departure'], leg['Arrival'], price, carrier + flight_num))

    return F_v, flights_v


def get_ticket_prices( origin_place  : str
                     , dest_place    : str
                     , outbound_date : datetime.date
                     , include_carriers         = None
                     , cabinclass    : str      = 'Economy'
                     , adults        : int      = 1
                     , use_cache     : bool     = False
                     , insert_into_livedb :bool = False
                     , host : str               = 'localhost') -> Union[None, Tuple]:
    """
    Returns the ticket prices for a flight

    :param origin_place: IATA code of the origin airport 'SIN'
    :param dest_place: IATA code of the destination airport 'KUL'
    :param outbound_date: outbound date # TODO: remove: in dash format '2017-02-15'
    :param include_carriers: IATA code of a _SINGLE_ airline code
    :param cabinclass: cabin class of the flight ticket (one of 'Economy', 'Business')
    :param adults: number of adult tickets booked
    :param: use_cache: bool indicator to signal to SkyScanner api to use cache.
    :param insert_into_livedb: indicator whether to insert the fetched flight into the livedb
    :param host: mysql host server name.
    :returns: None if no results, otherwise a tuple of forward prices, and itineraries.
    """

    # first check the local database
    flights_live_local = """
    SELECT as_of, flight_id, dep_date, dep_time, arr_date, price, carrier, flight_nb 
    FROM flights_live 
    WHERE orig = '{0}' AND dest = '{1}' AND dep_date = '{2}' AND cabin_class = '{3}' AND as_of > '{4}' 
    """.format(origin_place, dest_place, outbound_date, cabinclass, (datetime.datetime.now() - livedb_delay).isoformat())

    if include_carriers is not None:
        flights_live_local += "  AND carrier = '{0}'".format(include_carriers)  # include_carriers is only 1 in this case

    with MysqlConnectorEnv(host=host) as mysql_conn:
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
    if not result:  # nothing out
        return None

    # results are not empty
    F_v, flights_v = extract_Fv_flights_from_results(result)

    if insert_into_livedb:  # insert obtained flights into livedb
        insert_into_flights_live(origin_place, dest_place, flights_v, cabinclass)

    return F_v, flights_v


def reorganize_ticket_prices(itin) -> dict:
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
    """

    # get the days from the list of
    dep_day_hour = [(x[1].split('T'), x[2].split('T'), x[3], x[0], x[4]) for x in itin]

    reorg_tickets = dict()

    for (date_dt, hour), (arr_date, arr_hour), price, flight_id, flight_num in dep_day_hour:
        time_of_day_res = get_tod(hour)

        # part to insert into dict d
        if date_dt not in reorg_tickets.keys():
            reorg_tickets[date_dt] = dict()
        if time_of_day_res not in reorg_tickets[date_dt].keys():
            reorg_tickets[date_dt][time_of_day_res] = dict()
        # TODO: True is to follow the flights in the app
        reorg_tickets[date_dt][time_of_day_res][hour] = (flight_id, date_dt, hour, arr_date, arr_hour, price, flight_num, True)

    return reorg_tickets


def get_all_carriers( origin_place  : str
                    , dest_place    : str
                    , outbound_date : datetime.date
                    , cabinclass    = 'Economy') -> set:
    """ Gets all carriers for a selected route and selected date (direct flights only)

    :param origin_place: IATA code of the origin airport
    :param dest_place:   IATA code of the destination airport
    :param outbound_date: date of the flights between origin, destination
    :returns: flights for the route selected.
    """

    _, flights = get_ticket_prices( origin_place  = origin_place
                                  , dest_place    = dest_place
                                  , outbound_date = outbound_date
                                  , cabinclass    = cabinclass)

    return set([ flight[4][:2] for flight in flights])  # carrier names, a set
