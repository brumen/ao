# Module for checking that the origin is valid

import pandas as pd

from mysql_connector_env import MysqlConnectorEnv
# old get_carrier function
from ao_scripts.get_data import validate_airport
# from ao_codes            import iata_codes_airlines

from iata.codes import get_city_code, get_city_name


# iata_cities/airlines codes
#from ao_codes import iata_cities_codes   as IATA_CITIES_CODES,\
#                     iata_codes_cities   as IATA_CODES_CITIES,\
#                     iata_airlines_codes as IATA_AIRLINES_CODES,\
#                     iata_codes_airlines as IATA_CODES_AIRLINES


#IATA_CODES_CITIES_l   = list(IATA_CODES_CITIES.keys())
#IATA_CITIES_CODES_l   = list(IATA_CITIES_CODES.keys())
#IATA_AIRLINES_CODES_l = list(IATA_AIRLINES_CODES.keys())
#IATA_CODES_AIRLINES_l = list(IATA_CODES_AIRLINES.keys())


def show_airport_l(airport_partial_name: str) -> tuple:
    """
    Returns the list of airports from term.

    :param airport_partial_name: Partial name of the airport one is searching for.
    :returns: the list of potential candidates for the airport
    """

    airport_name_upper = airport_partial_name.upper()

    all_codes  = get_city_code(airport_name_upper)
    all_cities = get_city_name(airport_name_upper)

    # check if we get any cities or codes
    if all_codes:
        if all_cities:
            ret_cand = all_cities + all_codes
        else:
            ret_cand = all_codes

    elif all_cities:
        ret_cand = all_cities

    else:
        ret_cand = None

    if not ret_cand:  # nothing found, report problem
        return None

    if len(ret_cand) < 10:
        return ret_cand

    return ret_cand[:10]


def get_carrier_l(origin, dest):
    """
    Populates it w/ all carriers 

    """

    # get the three letter codes from origin, dest

    return IATA_AIRLINES_CODES_l + IATA_CODES_AIRLINES_l

    
def get_carrier_list_from_params(origin : str, dest :str):
    """ Gets the list from the params database.

    :param origin: origin IATA code destination
    :param dest: destination IATA code.
    """

    # get the three letter codes from origin, dest
    origin_upper, origin_valid = validate_airport(origin)
    dest_upper  , dest_valid   = validate_airport(dest  )

    if not origin_valid or not dest_valid:
        return None

    with MysqlConnectorEnv() as mconn:
        df1 = pd.read_sql_query( "SELECT DISTINCT(carrier) FROM params WHERE orig = '{0}' AND dest = '{1}'".format(origin_upper, dest_upper)
                               , mconn)

    ret_cand_1 = list(df1['carrier'])
    if len(ret_cand_1) == 0:  # no result
        return None

    # we have to return all the candidates
    # extend with flight names
    ret_cand_1.extend([iata_codes_airlines[x] for x in ret_cand_1])
    return ret_cand_1
