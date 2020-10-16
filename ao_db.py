# Air Option database construction functions

import time
import logging
import datetime
from typing import List, Tuple

# ao modules
import ao.ao_codes as ao_codes
import ao.ds       as ds

from ao.air_search import get_itins
from ao.mysql_connector_env import MysqlConnectorEnv


logger = logging.getLogger(__name__)


def run_db_mysql(s : str, host = 'localhost') -> List:
    """ Runs the query on the mysql database,

    :param s: query to be executed
    :param host: mysql db host.
    :returns: list of results obtained.
    """

    with MysqlConnectorEnv(host=host) as new_mysql:
        my_cursor = new_mysql.cursor()
        my_cursor.execute(s)

        return my_cursor.fetchall()


def find_dep_hour_day( dep_hour : str
                     , dep_day  : str ) -> tuple :
    """ Returns beg., end hours for dep_hours and dep_day

    :param dep_hour: one of 'morning', 'afternoon', 'evening', 'night'
    :param dep_day:  one of 'weekday', 'weekend'
    :returns: a tuple with first elt being the time of day TODO
    """

    dof_l = [0, 1, 2, 3, 4] if dep_day == 'weekday' else [5, 6]  # weekend

    # hr: stands for hour_range, s for start, e for end
    if dep_hour == 'morning':
        return ao_codes.morning_dt, dof_l

    if dep_hour == 'afternoon':
        return ao_codes.afternoon_dt, dof_l

    if dep_hour == 'evening':
        return ao_codes.evening_dt, dof_l

    if dep_hour == 'night':
        return ao_codes.night_dt, dof_l


def find_dep_hour_day_inv( dep_date : datetime.date
                         , dep_time : datetime.time ) -> Tuple:
    """ Inverse of the function above: computes hour_start, hour_end and day_of_week list from dep_date, dep_time.

    :param dep_date: departure date
    :param dep_time: departure time
    :returns:   month, dayofweek of the date/time
    """

    dof = 'weekday' if dep_date in ao_codes.weekday_days else 'weekend'

    if ao_codes.morning_dt[0] <= dep_time < ao_codes.morning_dt[1]:
        return 'morning', dof

    if ao_codes.afternoon_dt[0] <= dep_time < ao_codes.afternoon_dt[1]:
        return 'afternoon', dof

    if ao_codes.evening_dt[0] <= dep_time < ao_codes.evening_dt[1]:
        return 'evening', dof

    return 'night', dof

    
def update_flights_w_regs():
    """
    fixes the column reg_id in table flights 

    """

    update_str = """
    UPDATE flights 
    SET reg_id = {0} 
    WHERE (month = {1} AND tod = '{2}' and weekday_ind = '{3}')
    """

    with MysqlConnectorEnv() as mysql_conn:

        mysql_c = mysql_conn.cursor()
        mysql_c.execute('SELECT reg_id, month, tod, weekday_ind FROM reg_ids')

        with MysqlConnectorEnv() as mysql_conn_local:

            for row in mysql_c:

                update_str_curr = update_str.format(row[0], row[1], row[2], row[3])
                # TODO: THIS IS REALLY SLOW - FIND A BETTER WAY
                mysql_conn_local.cursor().execute(update_str_curr)
                mysql_conn_local.commit()


def find_location(loc_id, flights):
    """
    Finds the airport location as a string from loc_id (which is ID)

    """
    return [x['Code']
            for x in flights['Places']
            if x['Id'] == loc_id][0]


def accumulate_flights( origin_place
                      , dest_place
                      , outbound_date
                      , includecarriers = None
                      , acc_flight_l    = []
                      , curr_depth      = 0
                      , depth_max       = 1 ):
    """
    Insert flights into db.

    :param
    :param depth: current depth of the recursive search
    :param depth_max: maximum depth of the search
    """

    if curr_depth >= depth_max:
        return acc_flight_l

    # fetch flights
    flights = get_itins( origin_place    = origin_place
                       , dest_place      = dest_place
                       , outbound_date   = outbound_date
                       , includecarriers = includecarriers
                       , adults          = 1 )

    segments = flights['Segments']
    if not segments:  # if segments is empty
        return acc_flight_l

    lt      = time.localtime()
    itins   = flights['Itineraries']
    legs    = flights['Legs'       ]

    dest_id = int(flights['Query']['DestinationPlace'])
    orig_id = int(flights['Query']['OriginPlace'     ])

    orig    = find_location(orig_id, flights)
    dest    = find_location(dest_id, flights)

    as_of = str(lt.tm_year) + '-' + str(ds.d2s(lt.tm_mon)) + '-' + str(ds.d2s(lt.tm_mday)) + 'T' + \
            str(ds.d2s(lt.tm_hour)) + ':' + str(ds.d2s(lt.tm_min)) + ':' + str(ds.d2s(lt.tm_sec))

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
                logger.debug('Considering flight ', indiv_flights)
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
                            logger.debug(" ".join(["Inserting flight from"
                                                  , leg_orig_2
                                                  , "to"
                                                  , leg_dest_2
                                                  , "on"
                                                  , dep_date_2]))

                            acc_flight_l.extend(accumulate_flights( leg_orig_2
                                                                  , leg_dest_2
                                                                  , ds.convert_datedash_date(dep_date_2)
                                                                  , includecarriers = None
                                                                  , acc_flight_l    = []  # THIS HERE IS ABSOLUTELY NECC.
                                                                  , curr_depth      = curr_depth + 1
                                                                  , depth_max       = depth_max ) )

    return acc_flight_l


def commit_flights_to_live(flights_l : List):
    """
    Inserts flights_l into live table.

    :param flights_l: list of (as_of, orig, dest, dep_date, arr_date, carrier, price, outbound_leg_id)
                      types of these are:

    :returns: None
    """

    with MysqlConnectorEnv() as conn:  # odroid db.
        conn.cursor().executemany(
            """INSERT INTO flights_live(as_of, orig, dest, price, flight_id, dep_date, dep_time,
                                        arr_date, carrier, flight_nb, cabin_class)
               VALUES( %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s )"""
                                 , [( as_of, orig, dest, price, outbound_leg_id,
                                      dep_date.split('T')[0]
                                    , dep_date.split('T')[1]
                                    , arr_date, carrier, flight_nb, 'economy')
                                    for (as_of, orig, dest, dep_date, arr_date, carrier, price, outbound_leg_id, flight_nb)
                                    in flights_l ] )
        conn.commit()


def commit_flights_to_db( flights_l : List[datetime.date, str, str, datetime.date, datetime.date, str, float, str]
                        , host_db   = 'localhost' ) -> None:
    """
    Inserts into db the flights in the flights_l.

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
            dep_date_dt = ds.convert_datedash_date(dep_date)
            dep_time_dt = ds.convert_hour_time(dep_time)

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

        
def insert_flight( origin_place  : str
                 , dest_place    : str
                 , outbound_date : datetime.date
                 , includecarriers = None
                 , dummy           = False
                 , depth_max       = 0 ):
    """
    Finds flights between origin_place and dest_place on outbound_date and inserts them

    :param origin_place:    IATA value of the origin airport (like 'EWR')
    :param dest_place:      IATA value of dest. airport ('SFO')
    :param outbound_date:   date of the outbound flights one is trying to insert ('2016-10-28')
    :param includecarriers: airlines to use, can be None TODO: WHAT DOES THAT MEAN?
    :param dummy:           whether to acutally insert into the database, or just mock it
    :param depth:           recursive depth, i.e. if a flight is btw EWR -> ORD -> SFO,
                            if depth = 2 it will also check EWR -> ORD and ORD -> SFO
    :param direct_only:     whether to include only direct flights
    :param depth_max:
    :param existing_pairs:  TODO:
    """

    logger.debug(' '.join([ "Inserting flight from"
                          , origin_place
                          , "to"
                          , dest_place
                          , "on"
                          , outbound_date.isoformat() ]))

    flights_to_insert = accumulate_flights( origin_place
                                          , dest_place
                                          , outbound_date
                                          , includecarriers = includecarriers
                                          , acc_flight_l    = []
                                          , curr_depth      = 0
                                          , depth_max       = depth_max)

    if not dummy:
        commit_flights_to_live(flights_to_insert)


def ao_db_fill( dep_date_l : List[datetime.date]
              , dest_l     : List[str]
              , dummy      = False
              , depth_max  = 0 ):
    """
    Inserts into database all flights that can be found between
    IATA codes in the dep_date_l

    :param dep_date_l: list of departure dates, in the form datetime.date
    :param dest_l:     list of IATA codes destinations, could use as iata_codes.keys()
    :param depth_max:  depth of searches, i.e. EWR - SFO goes over DFW, then start again
                          from DFW, this shouldnt be larger than 2
    :type depth_max:   int
    :param dummy:      whether to do a database insert or just print out
    :type dummy:       bool
    :returns:          inserts into database what is found
    :rtype:            None
    """

    for dep_date in dep_date_l:
        for orig in dest_l:
            for dest in dest_l:
                print ("Inserting " + orig + " to " + dest + ".")
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
    """
    Performs the database maintanance: actions that should be undertaken

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
