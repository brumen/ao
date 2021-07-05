# Module for checking that the origin is valid

import pandas as pd

from typing import List, Union, Tuple

from ao.mysql_connector_env import MysqlConnectorEnv
from ao.iata.codes          import get_city_code, get_city_name


def get_airports(airport_partial_name: str) -> Union[Tuple, None]:
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

    
def get_carriers_on_route(origin : str, dest :str, host_db = 'localhost') -> List[str]:
    """ Gets the list from the params database.

    :param origin: origin IATA code destination ('EWR')
    :param dest: destination IATA code. ('SFO')
    :param host_db: name of host db connection, like 'localhost'
    :returns: list of IATA codes of airlines that fly that route
    """

    with MysqlConnectorEnv(host=host_db) as mconn:
        return list(pd.read_sql_query( "SELECT DISTINCT(carrier) FROM params WHERE orig = '{0}' AND dest = '{1}'".format(origin.upper(), dest.upper())
                                     , mconn )['carrier'])
