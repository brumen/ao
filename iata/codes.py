# To handle IATA retrival from the database

from mysql_connector_env import MysqlConnectorEnv


def get_airline_code( airline_name : str
                    , host         = 'localhost' ):
    """ Returns all the airline codes associated w/ airline

    :param airline: partial airline name, can be Adria, or ria.
    :param host: computer host where the database is.
    """

    with MysqlConnectorEnv(host=host) as connection:
        iata_c = connection.cursor()
        iata_c.execute("SELECT iata_code FROM iata_codes WHERE airline_name LIKE '%{0}%'".format(airline_name))

        all_codes = iata_c.fetchall()
        if not all_codes:  # all_codes is None
            return None

        return [airline_code[0] for airline_code in all_codes]


def get_airline_name( iata_code : str
                    , host      = 'localhost'):

    with MysqlConnectorEnv(host=host) as connection:
        iata_c = connection.cursor()
        iata_c.execute("SELECT airline_name FROM iata_codes WHERE iata_code LIKE '%{0}%'".format(iata_code))

        all_airline_codes = iata_c.fetchall()
        if not all_airline_codes:
            return None

        return [airline_name[0] for airline_name in all_airline_codes]


def get_city_code(city_name : str
                 , host     = 'localhost' ):

    with MysqlConnectorEnv(host=host) as connection:
        iata_c = connection.cursor()
        iata_c.execute("SELECT city_code FROM iata_cities WHERE city_name LIKE '%{0}%'".format(city_name))

        all_city_codes = iata_c.fetchall()
        if not all_city_codes:
            return None

        return [city_code[0] for city_code in all_city_codes]


def get_city_name( city_code : str
                 , host      = 'localhost' ):

    with MysqlConnectorEnv(host=host) as connection:
        iata_c = connection.cursor()
        iata_c.execute("SELECT city_name FROM iata_cities WHERE city_code LIKE '%{0}%'".format(city_code))

        all_city_names = iata_c.fetchall()
        if not all_city_names:
            return None

        return [city_name[0] for city_name in all_city_names]
