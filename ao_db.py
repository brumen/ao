# air option database construction functions

import sqlite3
from   skyscanner.skyscanner import Flights
import time
import logging
import datetime

# typing imports
from typing import List

# asynchronous mysql
import asyncio
import aiomysql

import ao_codes
from   ao_codes              import iata_cities_codes, iata_airlines_codes,\
                                    COUNTRY, CURRENCY, LOCALE,\
                                    SQLITE_FILE,\
                                    DB_HOST, DB_USER
import air_search
import ds
from   mysql_connector_env   import MysqlConnectorEnv, make_pymysql_conn

# logger
logger = logging.getLogger(__name__)


# instructions for db admin

#     copy_sqlite_to_mysql_by_carrier(delete_flights_in_sqlite=True)
# 2. database administration: (in mysql, run mysql-workbench)
#      call calibrate_all() : calibrates all pairs, but does not write to params_new
#      call insert_calibrate_all_to_params_new(): calibrates data and inserts them into params_new
#      call compare_new_params(): compares parameters in params and params_new
#      call copy_params_into_old(): copy params into params_old
#      call delete_old_live_data(int hours_old): deletes from flights_
#             live data that are older than hours_old
#      call push_new_params_to_prod(): copies params_new to params;
#      call insert_flights_live_into_flights_ord(): insert flights from flights_live db into flights_ord, historical db
# 3. potentially delete sqlite db on rasp


def run_db(s):
    """
    run query s on the database
    """

    res = []

    with sqlite3.connect(SQLITE_FILE) as conn:
        c = conn.cursor()
        for row in c.execute(s):
            res.append(row)

    return res


def run_db_mysql(s):
    """
    runs the query on the mysql database,

    :param s:   query to be executed
    :type s:    string
    :returns:   result of the mysql executed
    :rtype:     List
    """

    with MysqlConnectorEnv() as new_mysql:
        my_cursor = new_mysql.cursor()
        my_cursor.execute(s)
        return my_cursor.fetchall()


def find_dep_hour_day( dep_hour
                     , dep_day ):
    """
    returns beg., end hours for dep_hours and dep_day

    :param dep_hour: one of 'morning', 'afternoon', 'evening', 'night'
    :type dep_hour:  string
    :param dep_day:  one of 'weekday', 'weekend'
    :type dep_day:   string
    """

    # hr: stands for hour_range, s for start, e for end
    if dep_hour == 'morning':
        hr_s, hr_e = ao_codes.morning_dt
    elif dep_hour == 'afternoon':
        hr_s, hr_e = ao_codes.afternoon_dt
    elif dep_hour == 'evening':
        hr_s, hr_e = ao_codes.evening_dt
    elif dep_hour == 'night':
        hr_s, hr_e = ao_codes.night_dt
        
    if dep_day == 'weekday':
        dof_l = [0, 1, 2, 3, 4]
    else:
        dof_l = [5, 6]  # weekend
        
    return (hr_s, hr_e), dof_l
    

def find_dep_hour_day_inv(dep_date, dep_time):
    """
    inverse of the function above 
    computes hour_start, hour_end and day_of_week list from 

    :param dep_time: departure in datetime

    :returns:   month, dayofweek,
    :rtype:     TODO: FIX HERE
    """
    if ao_codes.morning_dt[0] <= dep_time < ao_codes.morning_dt[1]:
        dod = 'morning'
    elif ao_codes.afternoon_dt[0] <= dep_time < ao_codes.afternoon_dt[1]:
        dod = 'afternoon'
    elif ao_codes.evening_dt[0] <= dep_time < ao_codes.evening_dt[1]:
        dod = 'evening'
    else:
        dod = 'night' 
            
    if dep_date in ao_codes.weekday_days:
        dof = 'weekday'
    else:
        dof = 'weekend'
            
    return dod, dof

    
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

    
def copy_sqlite_to_mysql_by_carrier( add_flight_ids           = True
                                   , delete_flights_in_sqlite = False):
    """
    copies the flights data from sqlite database on raspberry to mysql database 
    and at the same time updates the existing flights with month, dayofweek, hour 

    :param add_flight_ids: whether to add new ids to the flights, should be True by default
    """

    # sqlite on raspberry
    try:
        c_ao = sqlite3.connect(SQLITE_FILE).cursor()
    except:
        raise

    # before this is executed do the following:
    # 1. clone the database (copy, there is a better way)
    # 2. perform vacuum on the database (command vacuum;)
    # 3. delete from flights

    ins_l = []
    all_flights = "SELECT * FROM flights"  #  WHERE carrier = '%s'" % (carrier)
    # insert this list into 
    add_flights_str = """INSERT INTO flights_ord
                         (as_of, price, reg_id, flight_id) 
                         VALUES (%s, %s, %s, %s)"""

    find_rid_str = """
    SELECT reg_id from reg_ids 
    WHERE month = %s AND tod = '%s' AND weekday_ind = '%s'
    """

    find_fid_str = """
    SELECT flight_id from flight_ids 
    WHERE flight_id_long = '%s'
    """

    ins_new_fid_str = """
        INSERT INTO flight_ids (flight_id_long, 
            orig, dest, dep_date, arr_date, carrier) 
        VALUES (%s, %s, %s, %s, %s, %s);
    """

    # this for loop finds all the flight_ids not previously in the database
    if add_flight_ids:

        with MysqlConnectorEnv() as mysql_conn_fid_ins:
            # cursor for inserting the flight_ids
            mysql_c_fid_ins = mysql_conn_fid_ins.cursor()

            fids_new = set() # Set()
            fids_size = 0

            for row in c_ao.execute(all_flights):

                dep_date, dep_time = row[3].split('T')  # departure date/time

                if '+' not in dep_time:
                    # find flight_id
                    flight_id_long = row[7]
                    mysql_c_fid_ins.execute(find_fid_str % flight_id_long)

                    if len(mysql_c_fid_ins.fetchall()) == 0:  # nothing was found
                        fids_new.add((flight_id_long, row[1], row[2], row[3], row[4], row[5]))
                        fids_size += 1

                    if fids_size > 1000:
                        mysql_c_fid_ins.executemany(ins_new_fid_str, list(fids_new))
                        mysql_conn_fid_ins.commit()
                        fids_size = 0
                        fids_new.clear()

            mysql_c_fid_ins.executemany(ins_new_fid_str, list(fids_new))
            mysql_conn_fid_ins.commit()

    with MysqlConnectorEnv() as mysql_conn_fid_rid:
        mysql_c_fid_rid = mysql_conn_fid_rid.cursor()

        # this part inserts all the flight price data into the database
        for row in c_ao.execute(all_flights):
            dep_date, dep_time = row[3].split('T')  # departure date/time
            dep_date_dt = ds.convert_datedash_date(dep_date)
            if '+' not in dep_time:
                dep_time_dt = ds.convert_hour_time(dep_time)
                month = dep_date_dt.month
                dod, dof = find_dep_hour_day_inv(dep_date_dt, dep_time_dt)
                # finds the reg_id (this will always return only 1 reg_id)
                mysql_c_fid_rid.execute(find_rid_str % (month, dod, dof) )
                reg_id_curr = mysql_c_fid_rid.fetchone()[0]
                # find flight_id
                flight_id_long = row[7]
                mysql_c_fid_rid.execute(find_fid_str % flight_id_long)
                flight_id_curr = mysql_c_fid_rid.fetchone()[0]
                ins_l.append((row[0], row[6], reg_id_curr, flight_id_curr))

            if len(ins_l) > 10000:  # 1000 works for sure, maybe even larger
                mysql_c_fid_rid.executemany(add_flights_str, ins_l)
                ins_l = []
                mysql_conn_fid_rid.commit()

        # flush the remaining ins_l
        mysql_c_fid_rid.executemany(add_flights_str, ins_l)
        mysql_conn_fid_rid.commit()  # final thing

    # delete flights from sqlite db
    if delete_flights_in_sqlite:
        c_ao.execute("DELETE FROM flights")
        c_ao.close()
    

def find_location(loc_id, flights):
    """
    Finds the airport location as a string from loc_id (which is ID)

    """
    return [x['Code']
            for x in flights['Places']
            if x['Id'] == loc_id][0]


def insert_into_db( flights
                  , acc_flight_l = []
                  , dummy        = False
                  , curr_depth   = 0
                  , depth_max    = 3 ):
    """
    Insert flights into db.

    :param flights: flights object as generated
    :param direct_only: consider only direct flights 
    :param dummy: if True, don't insert into database, just print
    :param depth: current depth of the recursive search
    :param depth_max: maximum depth of the search
    """

    if (curr_depth >= depth_max) or (flights is None):  # only conduct the search in this case
        return acc_flight_l

    # if none of these two, continue

    segments = flights['Segments']
    if not segments:  # if segments is empty
        return acc_flight_l

    lt      = time.localtime()
    itins   = flights['Itineraries']
    legs    = flights['Legs']
    dest_id = int(flights['Query']['DestinationPlace'])
    orig_id = int(flights['Query']['OriginPlace'])
    orig    = find_location(orig_id, flights)
    dest    = find_location(dest_id, flights)

    as_of = str(lt.tm_year) + '-' + str(ds.d2s(lt.tm_mon)) + '-' + str(ds.d2s(lt.tm_mday)) + 'T' + \
            str(ds.d2s(lt.tm_hour)) + ':' + str(ds.d2s(lt.tm_min)) + ':' + str(ds.d2s(lt.tm_sec))

    for leg in legs:
        # checking if direct flights (accepts 1 or 0)
        flight_numbers = leg['FlightNumbers']
        if len(flight_numbers) == 1:  # direct flight
            direct_ind = True
        else:
            direct_ind = False
            # look for flights between the in-between places and insert only the direct ones

            # find origin and destination of all legs, and add those
            for indiv_flights in flight_numbers:
                # find this flight in segms
                print ('Considering flight ', indiv_flights)
                print ('Acc flights:', acc_flight_l)
                carrier_id_2 = indiv_flights['CarrierId'   ]
                flight_nb_2  = indiv_flights['FlightNumber']
                for seg_2 in segments:
                    if carrier_id_2 == seg_2['Carrier'] and flight_nb_2 == seg_2['FlightNumber']:  # we found it
                        try:
                            leg_orig_2 = find_location(seg_2['OriginStation'], flights)
                        except (KeyError, IndexError):
                            print ("Origin station not found, exiting")
                            break
                        try:
                            leg_dest_2 = find_location(seg_2['DestinationStation'], flights)
                        except (KeyError, IndexError):
                            print ("Destination station not found, exiting")
                            break
                        try:
                            dep_date_2_full = seg_2['DepartureDateTime']  # date in '2016-10-29' format
                        except (KeyError, IndexError):
                            print ("Departure date/time not found")
                            break
                        try:
                            dep_date_2 = seg_2['DepartureDateTime'].split('T')[0]
                        except (KeyError, IndexError):
                            print ("Departure time not found")
                            break
                        # TODO: this combination might exist in the database already
                        logger.debug(" ".join(["Inserting flight from"
                                                  , leg_orig_2
                                                  , "to"
                                                  , leg_dest_2
                                                  , "on"
                                                  , dep_date_2]))

                        acc_flight_l.extend(insert_into_db( air_search.get_itins( origin_place  = leg_orig_2
                                                                                , dest_place    = leg_dest_2
                                                                                , outbound_date = ds.convert_datedash_date(dep_date_2) )
                                                          , acc_flight_l = []
                                                          , dummy        = dummy
                                                          , curr_depth   = curr_depth + 1
                                                          , depth_max    = depth_max ) )

        # This legs is direct flight
        outbound_leg_id = leg['Id']
        # TODO: CHECK THIS: low_price = itin['PricingOptions'][0]['Price']
        dep_date = leg['Departure']  # dep date in format: '2018-03-05T12:23:15'
        arr_date = leg['Arrival']    # same format as above

        # TODO: WHAT HAPPENS IF THERE ARE MULTIPLE CARRIERS?
        carrier = ",".join([ [x['Code'] for x in flights['Carriers'] if x['Id'] == carrier_id][0]
                             for carrier_id in leg['Carriers'] ])

        # find price for the outbound leg one is searching
        # TODO: Does it have to be at least one price, to do [0] at the end??
        price = [itin['PricingOptions'][0]["Price"] for itin in itins if itin['OutboundLegId'] == outbound_leg_id][0]

        print ("DIRECT IND:" + str(direct_ind))
        if direct_ind:
            acc_flight_l.append((as_of, orig, dest, dep_date, arr_date, carrier, price, outbound_leg_id))

    return acc_flight_l


def insert_into_db_final():
    """
    This does the actual insertion into the db
    """

    commit_insert(as_of
                  , orig
                  , dest
                  , dep_date
                  , arr_date
                  , carrier
                  , price
                  , outbound_leg_id)

    if direct_only:
        if direct_ind:
            # ins_l.append((as_of, orig, dest, dep_date, arr_date, carrier, price, outbound_leg_id, direct_ind))
            # commits the insert with all the additional structures TO CORRECT TO CORRECT
            commit_insert(as_of
                          , orig
                          , dest
                          , dep_date
                          , arr_date
                          , carrier
                          , price
                          , outbound_leg_id)

            if existing_pairs is not None:
                existing_pairs.update([(as_of, orig, dest, dep_date)])
    else:
        ins_l.append((as_of, orig, dest, dep_date, arr_date, carrier, price, outbound_leg_id, direct_ind))

    if not dummy:  # actually write to database
        with MysqlConnectorEnv() as conn_ao:
            conn_ao.executemany( 'INSERT INTO flights VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)'
                               , ins_l)
            conn_ao.commit()

        logger.debug(str(as_of) + "," + orig + "," + dest + "," + dep_date + "\n")
    else:
        print ("Flights: {0}".format(ins_l[0]))


def commit_insert( as_of
                 , orig
                 , dest
                 , dep_date
                 , arr_date
                 , carrier
                 , price
                 , outbound_leg_id ):

    """
    Commits the flight into database

    :param as_of:    date when the flight was fetched
    :param orig:     IATA code of the origin airport ('EWR')
    :param dest:     IATA code of the dest.  airport ('SFO')
    :param dep_date: departure date, in the form "dateTtime" TODO: FIX THIS PART

    """

    # check if outbound_leg_id is in the database, and return flight_id_used
    find_leg_str = "SELECT flight_id FROM flight_ids WHERE flight_id_long = '{0}'"
    reg_id_str = "SELECT reg_id FROM reg_ids WHERE month = {0} AND tod = '{1}' AND weekday_ind = '{2}'"

    with MysqlConnectorEnv() as mysql_conn:
        mysql_cur = mysql_conn.cursor()
        flight_leg_id = mysql_cur.execute(find_leg_str.format(outbound_leg_id)).fetchone()

        if flight_leg_id is None:  # nothing in flight_ids, insert flight_id into the flight_ids table
            mysql_cur.execute( "INSERT INTO flight_ids (flight_id_long, orig, dest, dep_date, arr_date, carrier)"
                             , (outbound_leg_id, orig, dest, dep_date, arr_date, carrier) )
            flight_id_used = mysql_cur.execute()  # TODO: FINISH HERE
        else:
            flight_id_used = ob_leg_res[0]

        # find reg_id
        dep_date, dep_time = dep_date.split('T')  # departure date/time
        dep_date_dt        = ds.convert_datedash_date(dep_date)
        dep_time_dt        = ds.convert_hour_time(dep_time)
        month              = dep_date_dt.month
        dod, dof           = find_dep_hour_day_inv(dep_date_dt, dep_time_dt)  # tod = dod, weekday_ind = dof

        mysql_cur.execute(reg_id_str.format(month, dod, dof))

    # insert into db
    logger.debug( ",".join([ str(as_of)
                           , orig
                           , dest
                           , dep_date
                           , outbound_leg_id]) + "\n" )

        
def insert_flight( origin_place  : str
                 , dest_place    : str
                 , outbound_date : datetime.date
                 , includecarriers = None
                 , adults          = 1
                 , dummy           = False
                 , depth_max       = 0 ):
    """
    Finds flights between origin_place and dest_place on outbound_date and inserts them

    :param origin_place: IATA value of the origin airport (like 'EWR')
    :param dest_place: IATA value of dest. airport ('SFO')
    :param outbound_date: date of the outbound flights one is trying to insert ('2016-10-28')
    :param includecarriers: airlines to use, can be None TODO: WHAT DOES THAT MEAN?
    :param adults:          how many adults to use
    :param dummy:           whether to acutally insert into the database, or just mock it
    :param depth:
    :param direct_only:     whether to include only direct flights
    :param depth_max:
    :param existing_pairs:  TODO:
    """

    logger.debug(" ".join([ "Inserting flight from"
                          , origin_place
                          , "to"
                          , dest_place
                          , "on"
                          , outbound_date.isoformat() ]))

    return insert_into_db( air_search.get_itins( origin_place    = origin_place
                                               , dest_place      = dest_place
                                               , outbound_date   = outbound_date
                                               , includecarriers = includecarriers
                                               , adults          = adults )
                         , dummy          = dummy
                         , curr_depth     = 0
                         , depth_max      = depth_max)


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
                except:  # catches all exception requests.HTTPError:
                    print ("Incorrect location values {0}, {1}".format(orig, dest))


def find_city_code(name_part):
    """
    Finds code of a city where name_part is part of the name

    :param name_part: part of the airline name that one is searching
    :type name_part:  string
    :returns:         list of airlines with that name
    :rtype:           list of strings
    """

    return [iata_cities_codes[city]
            for city in iata_cities_codes
            if name_part in city]


def find_airline_code(name_part):
    """
    Finds code of an airline where name_part is part of that name

    :param name_part: part of the airline name that one is searching
    :type name_part:  string
    :returns:         list of airlines with that name
    :rtype:           list of strings
    """

    return [iata_airlines_codes[airline]
            for airline in iata_airlines_codes
            if name_part in airline]


def perform_db_maintenance(action_list):
    """
    Performs the database maintanance.

    :param action_list: list of actions to perform
    :type action_list:  list of str
    :returns:           None
    :rtype:             None
    """

    with MysqlConnectorEnv() as mysql_conn:
        if "insert_live_flights_into_db" in action_list:
            mysql_conn.cursor().execute("CALL insert_flight_ids_from_flights_live\(\);")
            mysql_conn.cursor().execute("CALL insert_flights_live_into_flights_ord\(\);")


async def calibrate_all_2(loop, flight_ids_fixed, batch_size):
    """
    Calibrates the parameters with multiple processing

    """

    nb_flight_ids = len(flight_ids_fixed)
    print ("FINISHED preliminary ")
    pool = await aiomysql.create_pool( host     = 'localhost'
                                     , port     = 3306
                                     , user     = DB_USER
                                     , password = ao_codes.brumen_mysql_pass
                                     , db       = 'ao'
                                     , loop     = loop)

    curr_idx = 0
    while curr_idx < nb_flight_ids:
        batch_ids = flight_ids_fixed[curr_idx: (curr_idx+batch_size)]
        print ('BS', batch_ids[0], batch_ids[-1], '\n')

        batch_str = str(batch_ids)[1:-1]  # removing the [ and ]

        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.callproc('calibrate_inner', args=(batch_str, ))

        curr_idx += batch_size

    pool.close()
    await pool.wait_closed()


def run_calibrate_all_2(batch_size=1000):

    flight_ids = run_db_mysql('SELECT DISTINCT(flight_id) FROM flight_ids;');
    flight_ids_fixed = sorted([x[0] for x in flight_ids])

    loop = asyncio.get_event_loop()
    loop.run_until_complete(calibrate_all_2(loop, flight_ids_fixed, batch_size))

