# air option search and compute

import time
import datetime

from typing import Union, List, Dict, Optional

from requests              import ConnectionError
from skyscanner.skyscanner import Flights, FlightsCache
from sqlalchemy.orm.session import Session

from ao.ds                  import convert_date_datedash, d2s
from ao.ao_codes            import COUNTRY, CURRENCY, LOCALE, skyscanner_api_key, livedb_delay, get_tod
from ao.mysql_connector_env import MysqlConnectorEnv
from ao.flight              import create_session, Prices  # , FlightLive


# TODO: REMOVE THIS BELOW
class FlightLive:
    pass


def get_itins( origin          : str
             , dest            : str
             , outbound_date   : datetime.date
             , includecarriers : Union[List[str], None] = None
             , cabinclass      : str                    = 'Economy'
             , adults          : int                    = 1
             , use_cache       : bool                   = False
             , nb_tries        : int                    = 1
             , max_nb_tries    : int                    = 5 ) -> Union[Dict, None]:
    """ Returns itineraries for the selection from the Skyscanner API.

    :param origin: IATA code of the flight origin airport (e.g. 'SIN', or 'SFO')
    :param dest: IATA code of the flight destination airport (e.g. 'KUL', or 'EWR')
    :param outbound_date: date for flights to fetch
    :param includecarriers: IATA code of the airlines to use, if None, all airlines
    :param cabinclass: one of the following: Economy*, PremiumEconomy, Business, First
    :param adults: number of adults to get
    :param use_cache: whether to use Skyscanner cache for ticket pricer. This is not the local db, just cache part of Skyscanner
    :param nb_tries: number of tries that one tries to get a connection to SkyScanner
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
               if no flights could be found, return None
    """

    params_all = dict( country          = COUNTRY
                     , currency         = CURRENCY
                     , locale           = LOCALE
                     , originplace      = f'{origin}-sky'
                     , destinationplace = f'{dest}-sky'
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
            return get_itins( origin          = origin
                            , dest            = dest
                            , outbound_date   = outbound_date
                            , includecarriers = includecarriers
                            , cabinclass      = cabinclass
                            , adults          = adults
                            , nb_tries        = nb_tries + 1
                            , max_nb_tries    = max_nb_tries )

        return None  # this is handled appropriately in the get_ticket_prices


def find_carrier(carriers : List[str], carrier_id : str) -> Optional[str]:
    """ Finds the carrier from the ID list

    :param carriers:  list of carriers
    :param carrier_id: carrier one is searching for
    :returns:          Code of the carrier info if found, else None
    """

    for carrier_info in carriers:
        if carrier_id == carrier_info['Id']:
            return carrier_info['Code']

    return None  # None indicates failure


# def get_cached_flights(flights_in_ldb) -> Tuple:
#     """ Obtains the flight data from the local database
#
#     :param flights_in_ldb: flights as obtained from a local database.
#     :returns: F_v, flights_v_str
#     """
#
#     F_v = [x[5] for x in flights_in_ldb]
#     flights_v_str = []
#
#     for fl in flights_in_ldb:
#         # dep_date/hour
#         dep_date_str = fl[2].isoformat()
#         dep_time_str = (datetime.datetime.min + fl[3]).time().isoformat()  # converts to string
#         flights_v_str.append((fl[1],
#                               dep_date_str + 'T' + dep_time_str,
#                               fl[4].isoformat(), fl[5], fl[6] + fl[7]))
#
#     return F_v, flights_v_str


def insert_into_flights_live( origin : str
                            , dest   : str
                            , flights
                            , cabinclass ):
    """ Inserts the flights given in flights_v into the flights_live database

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


# def extract_prices_flights(result : Dict[str, Any]) -> Tuple:
#     """ Extracts the flight forward prices and flight data from the results provided
#
#     :param result: result of output from SkyScanner, dictionary structure:
#                           'Itineraries'
#                           'Currencies'
#                           'Agents'
#                           'Carriers'
#                           'Query'
#                           'Segments'
#                           'Places'
#                           'SessionKey'
#                           'Legs'
#                           'Status'
#     :returns:
#     """
#
#     F_v = []
#     flights_v = []
#
#     for itinerary, leg in zip(result['Itineraries'], result['Legs']):
#
#         flight_num_all = leg['FlightNumbers']
#
#         if len(flight_num_all) == 1:  # indicator if the flight is direct, the other test case is missing
#             carrier = find_carrier(result['Carriers'], leg['Carriers'][0])  # carriers = all carriers, leg['carriers'] are id of carrier
#             price = itinerary['PricingOptions'][0]['Price']  # TODO: THIS PRICE CAN BE DIFFERENT
#             flight_num = flight_num_all[0]['FlightNumber']  # TODO: HOW DO WE KNOW THAT WE HAVE THIS??
#             F_v.append(price)
#             # leg['Departure'] is departure date
#             flights_v.append((leg['Id'], leg['Departure'], leg['Arrival'], price, carrier + flight_num))
#
#     return F_v, flights_v


def get_ticket_prices( origin  : str
                     , dest    : str
                     , outbound_date : datetime.date
                     , include_carriers         = None
                     , cabinclass    : str      = 'Economy'
                     , adults        : int      = 1
                     , use_cache     : bool     = False
                     , insert_into_livedb :bool = False
                     , session       : Optional[Session] = None
                     , ) -> Union[None, List[FlightLive]]:
    """ Returns the list of live flights.

    :param origin: IATA code of the origin airport 'SIN'
    :param dest: IATA code of the destination airport 'KUL'
    :param outbound_date: outbound date # TODO: remove: in dash format '2017-02-15'
    :param include_carriers: IATA code of a _SINGLE_ airline code
    :param cabinclass: cabin class of the flight ticket (one of 'Economy', 'Business')
    :param adults: number of adult tickets booked
    :param: use_cache: bool indicator to signal to SkyScanner api to use cache.
    :param insert_into_livedb: indicator whether to insert the fetched flight into the livedb
    :param session: mysqlalchemy session, if None, one is made up directly in the function.
    :returns: None if no results, otherwise a list of FlightLive objects
    """

    session_used = session if session else create_session()

    flights_in_ldb = session_used.query(FlightLive)\
                                 .filter_by( orig       = origin
                                           , dest       = dest
                                           , dep_date   = outbound_date
                                           , cabinclass = cabinclass)\
                                 .filter(FlightLive.as_of > (datetime.datetime.now() - livedb_delay).isoformat() )

    if include_carriers is not None:
        flights_in_ldb = flights_in_ldb.filter_by(carrier=include_carriers)

    flights_in_ldb = flights_in_ldb.all()

    if flights_in_ldb:  # we have this in the database
        return flights_in_ldb

    # If flights not found in the cached db, continue with skyscanner fetch
    result = get_itins( origin          = origin
                      , dest            = dest
                      , outbound_date   = outbound_date
                      , includecarriers = include_carriers
                      , cabinclass      = cabinclass
                      , adults          = adults
                      , use_cache       = use_cache)

    # returns all one ways on that date
    if not result:  # nothing out
        return None

    # results are not empty
    # F_v = []
    # flights_v = []

    flights = []
    for itinerary, leg in zip(result['Itineraries'], result['Legs']):

        flight_num_all = leg['FlightNumbers']

        if len(flight_num_all) == 1:  # indicator if the flight is direct, the other test case is missing
            carrier = find_carrier(result['Carriers'], leg['Carriers'][0])  # carriers = all carriers, leg['carriers'] are id of carrier
            price = itinerary['PricingOptions'][0]['Price']  # TODO: THIS PRICE CAN BE DIFFERENT
            flight_num = flight_num_all[0]['FlightNumber']  # TODO: HOW DO WE KNOW THAT WE HAVE THIS??

            # leg['Departure'] is departure date
            flights_v.append((leg['Id'], leg['Departure'], leg['Arrival'], price, carrier + flight_num))

            # prices
            price = Prices( as_of     = now()
                          , price     = price
                          , reg_id    = TODO
                          , flight_id = carrier + flight_num )  # TODO: FIX THIS

            flights.append(FlightLive( as_of=now()
                                     , flight_id = leg['Id']
                                     , prices = Price()  # TODO - FIX HERE
                                     , dep_date = leg['Departure']  # leg['Departure'] is departure date
                                     , ) )


#     return F_v, flights_v
    if insert_into_livedb:  # insert obtained flights into livedb
        for flight in flights:
            session.add(flight)

        session.commit()

#         insert_into_flights_live(origin, dest, flights_v, cabinclass)

    return flights


def reorganize_ticket_prices(flights : List[FlightLive]) -> Dict[datetime.date, Dict[str, Dict[str, FlightLive]]]:
    """ Reorganize the flights by levels:
       day (datetime.date)
         time of day (morning, afternoon)
            hour (datetime.time)

        insert ((date, hour), (arr_date, arr_hour), price, flight_id) into dict d

    :param flights: Itinerary in the form of a list of ((u'2016-10-28', u'19:15:00'), 532.),
                   where the first is the departure date, second departure time, third flight price
    :returns: multiple level dictionary
    """

    # get the days from the list of
    # TODO: CHECK ITINERARIES IN THE UPSTREAM FUNCTION
    # dep_day_hour = [(x[1].split('T'), x[2].split('T'), x[3], x[0], x[4]) for x in itin]

    reorg_tickets = dict()

    # for (date_dt, hour), (arr_date, arr_hour), price, flight_id, flight_num in dep_day_hour:
    for flight in flights:
        hour = flight.as_of.time()
        time_of_day = get_tod(hour)
        date_  = flight.as_of.date()

        # part to insert into dict d
        if date_ not in reorg_tickets.keys():
            reorg_tickets[date_] = dict()

        if time_of_day not in reorg_tickets[date_].keys():
            reorg_tickets[date_][time_of_day] = dict()

        # TODO: MISSING STUFF
        price = None
        arr_hour = None
        reorg_tickets[date_][time_of_day][hour] = (flight.flight_id, date_, hour, flight.arr_date, arr_hour, price, flight.flight_id, True)

    return reorg_tickets


def get_all_carriers( origin        : str
                    , dest          : str
                    , outbound_date : datetime.date
                    , cabinclass    : str = 'Economy' ) -> set:
    """ Gets all carriers for a selected route and selected date (direct flights only)

    :param origin: IATA code of the origin airport
    :param dest:   IATA code of the destination airport
    :param outbound_date: date of the flights between origin, destination
    :param cabinclass: cabin class.
    :returns: flights for the route selected.
    """

    _, flights = get_ticket_prices( origin        = origin
                                  , dest          = dest
                                  , outbound_date = outbound_date
                                  , cabinclass    = cabinclass
                                  , )

    return set([ flight[4][:2] for flight in flights])  # carrier names, a set
