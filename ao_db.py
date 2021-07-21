# Air Option database construction functions

import time
import logging
import datetime

from typing import List, Tuple, Optional
from sqlalchemy.orm.session import Session

from ao.ds                  import d2s, convert_datedash_date, convert_hour_time
from ao.air_search          import get_itins
from ao.mysql_connector_env import MysqlConnectorEnv
from ao.flight              import Flight, create_session, Prices, AORegIds
from ao.ao_codes import ( weekday_days
                        , weekend_days
                        , morning
                        , afternoon
                        , evening
                        , night
                        , )

logger = logging.getLogger(__name__)


def insert_into_reg_ids_db(session : Session = None) -> None:
    """ Constructs table reg_ids, for historical reference

    :param session: sqlalchemy session for usage.
    """

    for dep_tod in ['morning', 'afternoon', 'evening', 'night']:
        for dep_weekday in ['weekday', 'weekend']:
            for dep_month in range(1, 13):  # months
                session.add(AORegIds(month=dep_month, tod=dep_tod, weekday_ind=dep_weekday))

    session.commit()


def find_dep_hour_day( dep_hour : str
                     , dep_day  : str ) -> tuple :
    """ Returns beg., end hours for dep_hours and dep_day

    :param dep_hour: one of 'morning', 'afternoon', 'evening', 'night'
    :param dep_day:  one of 'weekday', 'weekend'
    :returns: a tuple with first elt being the time of day TODO
    """

    dof_l = weekday_days if dep_day == 'weekday' else weekend_days

    # hr: stands for hour_range, s for start, e for end
    if dep_hour == 'morning':
        return morning, dof_l

    if dep_hour == 'afternoon':
        return afternoon, dof_l

    if dep_hour == 'evening':
        return evening, dof_l

    if dep_hour == 'night':
        return night, dof_l


def find_dep_hour_day_inv( dep_date : datetime.date
                         , dep_time : datetime.time ) -> Tuple:
    """ Inverse of the function above: computes hour_start, hour_end and day_of_week list from dep_date, dep_time.

    :param dep_date: departure date
    :param dep_time: departure time
    :returns:   month, dayofweek of the date/time
    """

    dof = 'weekday' if dep_date in weekday_days else 'weekend'

    if morning[0] <= dep_time < morning[1]:
        return 'morning', dof

    if afternoon[0] <= dep_time < afternoon[1]:
        return 'afternoon', dof

    if evening[0] <= dep_time < evening[1]:
        return 'evening', dof

    return 'night', dof


def find_location(loc_id : int, flights : List) -> str:
    """ Finds the airport location as a string from loc_id (which is ID).

    :param loc_id: location id searched over flights, loc_id is Skyscanner internal.
    :param flights: list of flights over which the location is searched.
    """

    return [place['Code'] for place in flights['Places'] if place['Id'] == loc_id][0]


def create_flight( as_of : datetime.date
                 , orig  : str
                 , dest  : str
                 , dep_date : datetime.date
                 , arr_date : datetime.date
                 , carrier  : str
                 , price    : float
                 , outbound_leg_id : int
                 , flight_nb       : str ) -> Flight:
    """ Creates a flight from the parameters given.

    :param as_of: as-of date
    :param orig: originating airport
    :param dest: destination airport.
    :param dep_date: departure date
    :param arr_date: arrival date
    :param carrier: carrier airline, like 'UA'
    :param price: price of the flight
    :param outbound_leg_id: outbound leg id.
    :param flight_nb: flight nb
    :returns: TODO
    """

    # first check if flight already exists in the database
    session = create_session()

    flights = session.query(Flight).filter_by(orig=orig, dest=dest, dep_date=dep_date, arr_date=arr_date, carrier=carrier).all()

    if flights:  # flights exist, just add prices

        # there should be only 1 flight
        assert len(flights) == 1, \
            f'Multiple flights corresponding to the same search params {orig}, {dest}, {dep_date}, {arr_date}, {carrier}.'

        flight = flights[0]

    else:
        flight = Flight( orig           = orig
                       , dest           = dest
                       , dep_date       = dep_date
                       , arr_date       = arr_date
                       , flight_id_long = outbound_leg_id)  # TODO: CHECK THESE STUFF

    price_o = Prices(as_of=as_of, price=price, reg_id=TODO, flight_id = flight.flight_id)  # TODO: CHECK HERE
    flight.prices.append(price_o)

    return flight


def accumulate_flights( origin          : str
                      , dest            : str
                      , outbound_date   : datetime.date
                      , includecarriers : Optional[List[str]] =  None
                      , acc_flight_l    : List                = []
                      , curr_depth      : int                 = 0
                      , depth_max       : int                 = 1 ):
    """ Get flights from Skyscanner and insert them into db. The function searches for flights recursively, i.e.
        TODO: describe here.

    :param origin: origin airport, e.g. 'SFO'
    :param dest: destination airport, e.g. 'EWR'
    :param outbound_date: outbound date for flights to be searched.
    :param includecarriers: Only consider carriers specified here. If None, consider all carriers.
    :param depth: current depth of the recursive search
    :param depth_max: maximum depth of the search
    """

    if curr_depth >= depth_max:
        return acc_flight_l

    # fetch flights
    flights = get_itins( origin          = origin
                       , dest            = dest
                       , outbound_date   = outbound_date
                       , includecarriers = includecarriers
                       , adults          = 1 )

    segments = flights['Segments']
    if not segments:  # if segments is empty
        return acc_flight_l

    lt      = time.localtime()
    itins   = flights['Itineraries']
    legs    = flights['Legs'       ]

    # destination and origin ID.
    dest_id = int(flights['Query']['DestinationPlace'])
    orig_id = int(flights['Query']['OriginPlace'     ])

    orig    = find_location(orig_id, flights)
    dest    = find_location(dest_id, flights)

    as_of = str(lt.tm_year) + '-' + str(d2s(lt.tm_mon)) + '-' + str(d2s(lt.tm_mday)) + 'T' + \
            str(d2s(lt.tm_hour)) + ':' + str(d2s(lt.tm_min)) + ':' + str(d2s(lt.tm_sec))

    for leg in legs:
        # checking if direct flights (accepts 1 or 0)
        flight_numbers = leg['FlightNumbers']
        if len(flight_numbers) == 1:  # direct flight
            # This legs is direct flight
            flight_nb = flight_numbers[0]['FlightNumber']  # there is only one flight here
            outbound_leg_id = leg['Id']
            # TODO: CHECK THIS: low_price = itin['PricingOptions'][0]['Price']
            dep_date = leg['Departure']  # dep date in format: '2018-03-05T12:23:15'
            arr_date = leg['Arrival']  # same format as above

            # TODO: WHAT HAPPENS IF THERE ARE MULTIPLE CARRIERS?
            carrier = ",".join([[x['Code'] for x in flights['Carriers'] if x['Id'] == carrier_id][0]
                                for carrier_id in leg['Carriers']])

            # find price for the outbound leg one is searching
            # TODO: Does it have to be at least one price, to do [0] at the end??
            price = [itin['PricingOptions'][0]["Price"]
                     for itin in itins
                     if itin['OutboundLegId'] == outbound_leg_id][0]

            acc_flight_l.append((as_of, orig, dest, dep_date, arr_date, carrier, price, outbound_leg_id, flight_nb))

        else:  # look for flights between the in-between places and insert only the direct ones
            # find origin and destination of all legs, and add those
            for indiv_flights in flight_numbers:
                # find this flight in segms
                logger.info(f'Considering flight {indiv_flights}')
                carrier_id_2 = indiv_flights['CarrierId'   ]
                flight_nb_2  = indiv_flights['FlightNumber']
                for seg_2 in segments:
                    if carrier_id_2 == seg_2['Carrier'] and flight_nb_2 == seg_2['FlightNumber']:  # we found it
                        try:
                            leg_orig_2      = find_location(seg_2['OriginStation'     ], flights)
                            leg_dest_2      = find_location(seg_2['DestinationStation'], flights)
                            dep_date_2      = seg_2['DepartureDateTime'].split('T')[0]
                        except (KeyError, IndexError):
                            logger.info("Origin, destination or departure dates not found.")
                            break
                        else:
                            # TODO: this combination might exist in the database already
                            logger.debug(f'Inserting flight from {leg_orig_2} to {leg_dest_2} on {dep_date_2}')

                            acc_flight_l.extend(accumulate_flights( leg_orig_2
                                                                  , leg_dest_2
                                                                  , convert_datedash_date(dep_date_2)
                                                                  , includecarriers = None
                                                                  , acc_flight_l    = []  # THIS HERE IS ABSOLUTELY NECC.
                                                                  , curr_depth      = curr_depth + 1
                                                                  , depth_max       = depth_max ) )

    return acc_flight_l


def commit_flights_to_db( flights_l : List[Tuple[datetime.date, str, str, datetime.date, datetime.date, str, float, str]]
                        , host_db   : str = 'localhost' ) -> None:
    """ Inserts into db the flights in the flights_l.

    :param flights_l: list of (as_of, orig, dest, dep_date, arr_date, carrier, price, outbound_leg_id)
                      types of these are:
    :param host_db: database host
    :returns: TODO: WHAT DOES IT DO HERE??/
    """

    with MysqlConnectorEnv(host=host_db) as conn:
        cur = conn.cursor()
        # first add all ids to the flights to insert
        # this will ignore duplicates on outbound_leg_id, which is a unique index
        cur.executemany( 'INSERT INTO flight_ids VALUES (?, ?, ?, ?, ?, ?)'
                       , [(outbound_leg_id, orig, dest, dep_date, arr_date, carrier)
                          for (as_of, orig, dest, dep_date, arr_date, carrier, price, outbound_leg_id)
                          in flights_l ] )
        conn.commit()

        # check if outbound_leg_id is in the database, and return flight_id_used
        find_leg_str = "SELECT flight_id FROM flight_ids WHERE flight_id_long = '{0}'"
        # find reg id from month, weekday, time of day
        reg_id_str   = "SELECT reg_id    FROM reg_ids    WHERE month = {0} AND tod = '{1}' AND weekday_ind = '{2}'"

        # this has to be done one by one
        for (as_of, orig, dest, dep_date, arr_date, carrier, price, outbound_leg_id) in flights_l:

            dep_date, dep_time = dep_date.split('T')  # departure date/time
            dep_date_dt = convert_datedash_date(dep_date)
            dep_time_dt = convert_hour_time(dep_time)

            time_of_day, weekday_ind = find_dep_hour_day_inv(dep_date_dt, dep_time_dt)

            # check reg id
            cur.execute(reg_id_str.format(dep_date_dt.month, time_of_day, weekday_ind))
            reg_id = cur.fetchone()

            cur.execute(find_leg_str.format(outbound_leg_id))
            flight_id = cur.fetchone()

            # as_of, price, reg_id, flight_id as_of_td
            cur.execute( """ INSERT INTO flights_ord 
                             (now(), {0}, {1}, {2}, td(now()))
                         """.format(price, reg_id, flight_id))

        conn.commit()

        
def insert_flight( origin        : str
                 , dest          : str
                 , outbound_date : datetime.date
                 , includecarriers = None
                 , dummy           = False
                 , depth_max       = 0 ):
    """
    Finds flights between origin_place and dest_place on outbound_date and inserts them

    :param origin: IATA value of the origin airport (like 'EWR')
    :param dest: IATA value of dest. airport ('SFO')
    :param outbound_date:   date of the outbound flights one is trying to insert ('2016-10-28')
    :param includecarriers: airlines to use, can be None TODO: WHAT DOES THAT MEAN?
    :param dummy: whether to acutally insert into the database, or just mock it
    :param depth_max: recursive depth, i.e. if a flight is btw EWR -> ORD -> SFO,
                  if depth = 2 it will also check EWR -> ORD and ORD -> SFO
    :param depth_max: depth to which the flights are searched TODO: EXPLAIN HERE.
    """

    logger.debug(' '.join([ 'Inserting flight from'
                          , origin
                          , 'to'
                          , dest
                          , 'on'
                          , outbound_date.isoformat() ]))

    flights_to_insert = accumulate_flights( origin
                                          , dest
                                          , outbound_date
                                          , includecarriers = includecarriers
                                          , acc_flight_l    = []
                                          , curr_depth      = 0
                                          , depth_max       = depth_max)

    if not dummy:
        commit_flights_to_live(flights_to_insert)


def insert_flights_into_db( dep_dates : List[datetime.date]
                          , dests     : List[str]
                          , dummy     : bool = False
                          , depth_max : int  = 0 ) -> None:
    """ Inserts into database all flights that can be found between IATA codes in the dep_date_l

    :param dep_dates: list of departure dates, in the form datetime.date
    :param dests: list of IATA codes destinations, could use as iata_codes.keys()
    :param depth_max: depth of searches, i.e. EWR - SFO goes over DFW, then start again
                          from DFW, this shouldnt be larger than 2
    :param dummy: whether to do a database insert or just print out
    :returns: inserts into database what is found,
    """

    for dep_date in dep_dates:
        for orig in dests:
            for dest in dests:
                logger.info('Inserting {0} to {1}'.format(orig, dest))

                if orig == dest:
                    break
                try:
                    insert_flight( orig
                                 , dest
                                 , dep_date
                                 , depth_max     = depth_max
                                 , dummy         = dummy )
                    insert_flight( dest
                                 , orig
                                 , dep_date
                                 , depth_max     = depth_max
                                 , dummy         = dummy )

                except Exception as e:
                    logger.info("Incorrect location values {0}, {1}".format(orig, dest))
                    logger.info('Error: {0}'.format(str(e)))


def perform_db_maintenance(action_list : List[str]
                          , host_db = 'localhost' ) -> None:
    """ Performs the database maintenance: actions that should be undertaken

    :param action_list: list of actions to perform:
                           insert_flights_live_into_flights_ord_prasic
                           calibrate_and_push_prasic
                           copy_params_to_odroid
                           copy_flights_live_odroid_to_prasic
    :param host_db: host database
    """

    if "insert_flights_live_into_flights_ord_prasic" in action_list:
        # inserts flights_live into (flights_ord & flight_ids) _ON THE SAME DATABASE_
        # TODO: NOT SURE IF THIS WORKS
        with MysqlConnectorEnv(host=host_db) as mysql_conn:
            mysql_conn.cursor().callproc('insert_flights_live')

    elif 'calibrate_and_push_prasic' in action_list:
        # calibrates the parameters and pushes them to params on the same database
        with MysqlConnectorEnv(host=host_db) as mysql_conn:
            mysql_conn.cursor().callproc('insert_calibrate_all_to_params_new')
            mysql_conn.cursor().callproc('copy_params_into_old'   )
            mysql_conn.cursor().callproc('push_new_params_to_prod')

    elif 'copy_params_to_odroid' in action_list:
        # copies parameters from prasic to odroid
        with MysqlConnectorEnv(host=host_db) as mysql_conn_calibrate, MysqlConnectorEnv() as mysql_conn_odroid:
            calibrate_cur = mysql_conn_calibrate.cursor()
            odroid_cur    = mysql_conn_odroid.cursor()
            odroid_cur.execute('DELETE FROM params')
            mysql_conn_odroid.commit()  # making sure it deletes parameters
            calibrate_cur.execute('SELECT * from params;')
            odroid_cur.executemany( 'INSERT INTO params VALUES (%s, %s, %s, %s, %s, %s, %s, %s)'
                                  , calibrate_cur.fetchall() )
            mysql_conn_odroid.commit()

    elif 'copy_flights_live_odroid_to_prasic' in action_list:
        # copies live flights from odroid to prasic
        with MysqlConnectorEnv(host=host_db) as mysql_conn_calibrate, MysqlConnectorEnv(host='odroid') as mysql_conn_odroid:
            calibrate_cur = mysql_conn_calibrate.cursor()
            odroid_cur    = mysql_conn_odroid.cursor()

            # this takes care if duplicates
            odroid_cur.execute('SELECT * from flights_live;')
            for row in odroid_cur:
                date_fetched, orig, dest, price, flight_nb, start_date, duration, end_date, airline, code, coach_class = row 
                # row in the form: 
                # datetime.datetime(2018, 5, 11, 11, 34, 35), 'BDL', 'FLL', 101.98, '9796-1806191642--32171-0-11560-1806191950', datetime.date(2018, 6, 19), datetime.timedelta(0, 60120), datetime.datetime(2018, 6, 19, 19, 50), 'B6', '1459', 'economy')
                start_date_tmp = datetime.datetime(start_date.year, start_date.month, start_date.day)
                calibrate_cur.execute( "INSERT INTO flights_live VALUES ('" + \
                                           "','".join([ date_fetched.isoformat()
                                                        , orig
                                                        , dest
                                                        , str(price)
                                                        , flight_nb
                                                        , start_date.isoformat()
                                                        , (start_date_tmp + duration).time().isoformat()
                                                        , end_date.isoformat()
                                                        , airline
                                                        , code
                                                        , coach_class]) + "')")
            # old routines 
            # calibrate_cur.executemany( """INSERT INTO flights_live 
            #                               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
            #                          , odroid_cur.fetchall() )
            mysql_conn_calibrate.commit()

            # remove the live flights from odroid
            odroid_cur.execute('DELETE FROM flights_live;')
            mysql_conn_odroid.commit()


# certain other maintanance operations:
#   : to crate a schema and functions on the target database
#     run on prasic.local:  mysqldump --routines -u brumen -d ao | mysql -h odroid.local -u brumen -p  ao
#   : to copy table reg_ids between databases
#     run on prasic.local:  mysqldump -u brumen ao reg_ids | mysql -h odroid.local -u brumen -p  ao
