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
