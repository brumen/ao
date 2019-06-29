# To handle IATA retrival from the database

from mysql_connector_env import MysqlConnectorEnv


def get_airline_coda(airline):

    with MysqlConnectorEnv() as connection:
        iata_c = connection.cursor()
        iata_c.execute("SELECT iata_code FROM iata_codes WHERE airline_name = '{0}'".format(airline))

        return iata_c.fetchone()[0]


def get_city_id(city):

    with MysqlConnectorEnv() as connection:
        iata_c = connection.cursor()
        iata_c.execute("SELECT city_id FROM iata_cities WHERE city = '{0}'".format(city))

        return iata_c.fetchone()[0]
