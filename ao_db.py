# air option database construction functions

import sqlite3
from   skyscanner.skyscanner import Flights
import time
import logging

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
# 1. Copy from sqlite db on raspberry into mysql database on prasic:
#     first must mount /home/brumen/rasp
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


def create_ao_db():
    """
    creates the sqlite3 database for collecting Flights

    """

    create_flights = """
            CREATE TABLE flights 
                (as_of TEXT, orig TEXT, dest TEXT, 
                 dep_date TEXT, arr_date TEXT, 
                 carrier TEXT, price REAL,
                 id TEXT, direct_flight INTEGER)
    """

    with sqlite3.connect(SQLITE_FILE) as conn:
        c = conn.cursor()
        c.execute(create_flights)
        conn.commit()  # TODO: DO YOU NEED THIS


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
    if dep_time >= ao_codes.morning_dt[0] and dep_time < ao_codes.morning_dt[1]:
        dod = 'morning'
    elif dep_time >= ao_codes.afternoon_dt[0] and dep_time < ao_codes.afternoon_dt[1]:
        dod = 'afternoon'
    elif dep_time >= ao_codes.evening_dt[0] and dep_time < ao_codes.evening_dt[1]:
        dod = 'evening'
    else:
        dod = 'night' 
            
    if dep_date in ao_codes.weekday_days:
        dof = 'weekday'
    else:
        dof = 'weekend'
            
    return dod, dof

    
def insert_into_reg_ids_db():
    """
    populates table reg_ids, for historical reference 

    """

    ins_l = []
    for dep_hour in ['morning', 'afternoon', 'evening', 'night']:
        for dep_day in ['weekday', 'weekend']:
            for dep_season in range(1,13):
                ins_l.append((dep_season, dep_hour, dep_day))
    add_regs_str = """INSERT INTO reg_ids
                      (month, tod, weekday_ind)
                      VALUES (%s, %s, %s)
    """

    with MysqlConnectorEnv() as mysql_conn:
        mysql_conn.cursor().executemany(add_regs_str, ins_l)
        mysql_conn.commit()


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

        with make_pymysql_conn() as mysql_conn_fid_ins:
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

    with make_pymysql_conn() as mysql_conn_fid_rid:
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
    

def insert_into_itin( originplace      = 'SIN-sky'
                    , destinationplace = 'KUL-sky'
                    , date_today       = '2016-08-25'
                    , outbounddate     = '2016-10-28'
                    # inbounddate='2016-10-31',
                    , country          = COUNTRY
                    , currency         = CURRENCY
                    , locale           = LOCALE
                    , includecarriers  = 'SQ'
                    , adults           = 1):
    """
    inserts itiniraries into the database 
    uses sqlite db on the raspberry pi

    """

    flights_service = Flights(ao_codes.skyscanner_api_key)
    result = flights_service.get_result( country          = country
                                       , currency         = currency
                                       , locale           = locale
                                       , originplace      = originplace
                                       , destinationplace = destinationplace
                                       , outbounddate     = outbounddate
                                       # inbounddate = inbounddate,
                                       , includecarriers  = includecarriers
                                       , adults           = adults).parsed

    with sqlite3.connect(SQLITE_FILE) as conn:
        c = conn.cursor()
        # returns all one ways on that date
        ri = result['Itineraries']
        for itin in ri:
            po = itin['PricingOptions']
            for po_elt in po:
                price = po_elt['Price']
                ins_str = "INSERT INTO itins VALUES ('{0}', '{1}', '{2}', '{3}', '{4}', {5})"\
                          .format( date_today
                                 , outbounddate
                                 , originplace
                                 , destinationplace
                                 , includecarriers
                                 , price)
                c.execute(ins_str)
        conn.commit()


def find_location(loc_id, flights):
    """
    finds the airport location as a string from loc_id (which is ID)

    """
    return [x['Code']
            for x in flights['Places']
            if x['Id'] == loc_id][0]


def insert_into_db( flights
                  , direct_only     = False
                  , dummy           = False
                  , depth           = 0
                  , depth_max       = 0
                  , use_cache       = False
                  , existing_pairs  = None):
    """
    insert flights into db, uses sqlite on rasp pi

    :param flights: flights object as generated
    :param direct_only: consider only direct flights 
    :param dummy: if True, don't insert into database, just print
    :param depth: depth of the recursive search
    """

    if flights is None:
        return

    segms = flights['Segments']
    if segms == []:
        return 

    lt      = time.localtime()
    itins   = flights['Itineraries']
    legs    = flights['Legs']
    dest_id = int(flights['Query']['DestinationPlace'])
    orig_id = int(flights['Query']['OriginPlace'])
    orig    = find_location(orig_id, flights)
    dest    = find_location(dest_id, flights)

    as_of = str(lt.tm_year) + '-' + str(ds.d2s(lt.tm_mon)) + '-' + str(ds.d2s(lt.tm_mday)) + 'T' + \
            str(ds.d2s(lt.tm_hour)) + ':' + str(ds.d2s(lt.tm_min)) + ':' + str(ds.d2s(lt.tm_sec))

    ins_l = []
    for leg in legs:
        # checking if direct flights (accepts 1 or 0)
        if len(leg['FlightNumbers']) == 1:  # direct flight
            direct_ind = 1
        else:
            direct_ind = 0
            # look for flights between the in-between places and insert only the direct ones
            if depth < depth_max:  # only conduct the search in this case
                # find origin and destination of all legs, and add those
                for indiv_flights in leg['FlightNumbers']:
                    # find this flight in segms
                    carrier_id_2 = indiv_flights['CarrierId']
                    flight_nb_2 = indiv_flights['FlightNumber']
                    for seg_2 in segms:
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
                            # check if this combination exists in the database already (THIS CAN BE OPTIMIZED)
                            # check_exists_str = """
                            # SELECT COUNT(*) FROM flights WHERE as_of = '%s' AND orig = '%s' AND dest = '%s' AND dep_date = '%s'
                            # """ %(as_of, leg_orig_2, leg_dest_2, dep_date_2_full)
                            # res = c_ao.execute(check_exists_str).next()[0]
                            # if res == 0:
                            #if (as_of, leg_orig_2, leg_dest_2, dep_date_2_full) not in existing_pairs:
                            #    # update the existing pairs
                            insert_flight(origin_place     = leg_orig_2,
                                          dest_place       = leg_dest_2,
                                          outbound_date    = dep_date_2,
                                          includecarriers  = None,
                                          adults           = 1,
                                          dummy            = dummy,
                                          depth            = depth+1,
                                          direct_only      = direct_only,
                                          depth_max        = depth_max,
                                          use_cache        = use_cache)
            
        outbound_leg_id = leg['Id']
        # low_price = itin['PricingOptions'][0]['Price']
        dep_date = leg['Departure']
        arr_date = leg['Arrival']
        carrier_id_l = leg['Carriers']  # [0] (multiple carriers, list)
        carrier_l = []
        for carrier_id in carrier_id_l:
            carrier_used = [x['Code'] for x in flights['Carriers'] if x['Id'] == carrier_id][0]
            carrier_l.append(carrier_used)
        carrier = ",".join(carrier_l)
        # find price (BADLY PROGRAMMED)
        for itin in itins:
            itin_leg_id = itin['OutboundLegId']
            if itin_leg_id == outbound_leg_id:
                # get pricing
                price = itin['PricingOptions'][0]["Price"]  # THIS PRICE CAN BE DIFFERENT
                break
        if direct_only:
            if direct_ind == 1:
                # ins_l.append((as_of, orig, dest, dep_date, arr_date, carrier, price, outbound_leg_id, direct_ind))
                # commits the insert with all the additional structures TO CORRECT TO CORRECT 
                commit_insert( as_of
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

    if not dummy:  # TO BE FURTHER CORRECTED 
        with sqlite3.connect(SQLITE_FILE) as conn_ao:
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
    commits the flight into mysql db using mysql_conn, mysql_c is the cursor to the connection 

    """

    # check if outbound_leg_id is in the database, and return flight_id_used
    find_ob_leg_str = "SELECT flight_id FROM flight_ids WHERE flight_id_long = '%s'"
    find_ob_leg_id = mysql_c.execute(find_ob_leg_str % outbound_leg_id)
    ob_leg_res = mysql_c.fetchone()
    if ob_leg_res is None:  # nothing in flight_ids, insert flight_id into the flight_ids table 
        # mysql_c.execute("INSERT INTO flight_ids (flight_id_long, orig, dest, dep_date, arr_date, carrier)",
        #                (outbound_leg_id, orig, dest, dep_date, arr_date, carrier))
        g = 1  # TODO: REMOVE THIS LATER
    else:
        flight_id_used = ob_leg_res[0]

    # find reg_id
    dep_date, dep_time = dep_date.split('T')  # departure date/time
    dep_date_dt = ds.convert_datedash_date(dep_date)
    dep_time_dt = ds.convert_hour_time(dep_time)
    month = dep_date_dt.month
    dod, dof = find_dep_hour_day_inv(dep_date_dt, dep_time_dt)  # tod = dod, weekday_ind = dof
    reg_id_str = "SELECT reg_id FROM reg_ids WHERE month = %s AND tod = '%s' AND weekday_ind = '%s'"
    mysql_c.execute(reg_id_str % (month, dod, dof))

    # insert into db
    logger.debug( ",".join([ str(as_of)
                           , orig
                           , dest
                           , dep_date
                           , outbound_leg_id]) + "\n" )
    

def insert_into_db_cache( flights
                        , dummy          = False
                        , existing_pairs = None):
    """
    insert flights into db from Skyscanner cache, a much simplified version of above

    :param flights:     flights object as generated
    :param direct_only: consider only direct flights 
    :param conn_ao:     connection to ao
    :param c_ao: cursor to ao
    :param dummy: if True, don't insert into database, just print
    :param depth: depth of the recursive search
    """

    existing_pairs_cache = existing_pairs  # existing_pairs: existing pairs in the database

    def find_location(loc_id, flights):
        """
        finds the airport location as a string from loc_id (which is ID)
        """
        return [x['IataCode'] for x in flights['Places'] if x['PlaceId'] == loc_id][0]

    def find_carrier(carrier_id, carriers):
        """
        finds the carrier from the carrier id
        """
        return [x['Name'] for x in carriers if x['CarrierId'] == carrier_id][0]
        
    if flights is None:
        return
    quotes = flights['Quotes']
    if quotes is None:
        return 
    carriers = flights['Carriers']
    
    ins_l = []
    for quote in quotes:  # iterate over flights 
        direct_ind = quote['Direct'] == True
        if direct_ind:
            as_of = quote['QuoteDateTime']  # date and time of a quote
            dest_id = int(quote['OutboundLeg']['DestinationId'])
            orig_id = int(quote['OutboundLeg']['OriginId'])
            orig = find_location(orig_id, flights)
            dest = find_location(dest_id, flights)
            carrier = find_carrier(quote['OutboundLeg']['CarrierIds'][0], carriers)  # this is fine as only direct flights considered
            dep_date = quote['OutboundLeg']['DepartureDate']
            min_price = quote['MinPrice']
            if (as_of, orig, dest, dep_date) not in existing_pairs_cache:
                ins_l.append((as_of, orig, dest, dep_date, carrier, min_price))
                if existing_pairs is not None:
                    existing_pairs_cache.update([(as_of, orig, dest, dep_date)])

    if not dummy:
        c_ao.executemany('INSERT INTO flights_cache VALUES (?, ?, ?, ?, ?, ?)', ins_l)
        conn_ao.commit()
    else:
        print ("Flights: {0}".format(ins_l))

        
def insert_flight(origin_place    = 'SIN',
                  dest_place      = 'KUL',
                  outbound_date   = '2016-10-28',
                  includecarriers = None,
                  adults          = 1,
                  dummy           = False,
                  depth           = 0,
                  direct_only     = True,
                  depth_max       = 0,
                  use_cache       = False,
                  existing_pairs  = None):

    logger.debug(" ".join([ "Inserting flight from"
                          , origin_place
                          , "to"
                          , dest_place
                          , "on"
                          , outbound_date]))

    flights = air_search.get_itins( origin_place    = origin_place
                                  , dest_place      = dest_place
                                  , outbound_date   = outbound_date
                                  , includecarriers = includecarriers
                                  , adults          = adults
                                  , use_cache       = use_cache)
    if not use_cache:
        insert_into_db( flights
                      , dummy          = dummy
                      , depth          = depth
                      , direct_only    = direct_only
                      , depth_max      = depth_max
                      , existing_pairs = existing_pairs )
    else:
        insert_into_db_cache( flights
                            , dummy          = dummy
                            , existing_pairs = existing_pairs)


def ao_db_fill( dep_date_l
              , dest_l
              , dummy     = False
              , depth_max = 0 ):
    """
    Inserts into database all flights that can be found between
    IATA codes in the dep_date_l

    :param dep_date_l: list of departure dates in the form 2016-10-28
    :type dep_date_l:  list of str
    :param dest_l:     list of IATA codes destinations, could use as iata_codes.keys()
    :type dest_l:      list of str
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
                    insert_flight( origin_place  = orig
                                 , dest_place    = dest
                                 , outbound_date = dep_date
                                 , depth_max     = depth_max
                                 , dummy         = dummy )
                    insert_flight( origin_place  = dest
                                 , dest_place    = orig
                                 , outbound_date = dep_date
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

