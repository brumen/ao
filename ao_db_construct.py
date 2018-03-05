# Historical functions that may be used for repair

import sqlite3
from   mysql_connector_env   import MysqlConnectorEnv
from   ao_codes              import SQLITE_FILE


def insert_into_reg_ids_db():
    """
    Constructs table reg_ids, for historical reference

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


def insert_into_itin( originplace      = 'SIN-sky'
                    , destinationplace = 'KUL-sky'
                    , date_today       = '2016-08-25'
                    , outbounddate     = '2016-10-28'
                    , includecarriers  = 'SQ'
                    , adults           = 1):
    """
    Inserts itiniraries into the database.

    """

    flights_result = Flights(ao_codes.skyscanner_api_key).get_result( country          = COUNTRY
                                                                    , currency         = CURRENCY
                                                                    , locale           = LOCALE
                                                                    , originplace      = originplace
                                                                    , destinationplace = destinationplace
                                                                    , outbounddate     = outbounddate
                                                                    , includecarriers  = includecarriers
                                                                    , adults           = adults).parsed
    # extract flights to add and
    flights_to_add = []
    for itinerary in flights_result['Itineraries']:
        for po_elt in itinerary['PricingOptions']:
            flights_to_add.append([ date_today
                                  , outbounddate
                                  , originplace
                                  , destinationplace
                                  , includecarriers
                                  , po_elt['Price'] ])

    with MysqlConnectorEnv() as conn:
        cursor = conn.cursor()
        cursor.executemany( "INSERT INTO itins VALUES ('%s', '%s', '%s', '%s', '%s', %s)"
                          , flights_to_add)


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
