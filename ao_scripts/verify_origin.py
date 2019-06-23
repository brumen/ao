# Module for checking that the origin is valid

# old get_carrier function
from ao_scripts.get_data import validate_airport
from ao_codes            import iata_codes_airlines
from mysql_connector_env import MysqlConnectorEnv

# iata_cities/airlines codes
from ao_codes import iata_cities_codes   as IATA_CITIES_CODES,\
                     iata_codes_cities   as IATA_CODES_CITIES,\
                     iata_airlines_codes as IATA_AIRLINES_CODES,\
                     iata_codes_airlines as IATA_CODES_AIRLINES


IATA_CODES_CITIES_l   = list(IATA_CODES_CITIES.keys())
IATA_CITIES_CODES_l   = list(IATA_CITIES_CODES.keys())
IATA_AIRLINES_CODES_l = list(IATA_AIRLINES_CODES.keys())
IATA_CODES_AIRLINES_l = list(IATA_CODES_AIRLINES.keys())


def is_valid_origin(origin : str) -> bool :
    """
    Returns true if origin airport is in a list of airports

    :param origin: IATA name, or full name of the origin airport
    :returns:      True if origin is in the list of airports
    """

    if origin is None:
        return False

    ret_cand_1 = list( filter( lambda x: x.startswith(origin.upper())
                             , IATA_CITIES_CODES_l + IATA_CODES_CITIES_l ) )

    return not (len(ret_cand_1) == 0)


def is_valid_airline(airline : str) -> bool :
    """
    Returns true if the airline is valid, otherwise false
    returns the list of places from term

    :param airline: IATA code of the airline to be searched
    :returns:       True if airline is valid, False if not
    """

    if airline is None:
        return False

    ret_cand_1 = list( filter( lambda x: x.upper().startswith(airline.upper())
                             , IATA_AIRLINES_CODES_l + IATA_CODES_AIRLINES_l ) )

    return not (len(ret_cand_1) == 0)


def show_airport_l(airport_partial_name: str) -> tuple:
    """
    Returns the list of airports from term.

    :param airport_partial_name: Partial name of the airport one is searching for.
    :returns: the list of potential candidates for the airport
    """

    ret_cand = [x for x in IATA_CITIES_CODES_l + IATA_CODES_CITIES_l
                if airport_partial_name.upper() in x ]

    if not ret_cand:  # nothing found, report problem
        return None

    if len(ret_cand) < 10:
        return ret_cand

    return ret_cand[:10]


def show_airline_l(airline_partial_name : str) -> tuple:
    """
    Returns the list of IATA airlines for the partial name searched.

    :param airline_partial_name: partial name of the airline one is searching for.
    :returns: list of names one for the airlines one is searching for.
    """

    ret_cand_1 = list(filter( lambda x: airline_partial_name.upper() in x.upper()
                            , IATA_AIRLINES_CODES_l + IATA_CODES_AIRLINES_l ) )

    if not ret_cand_1:  # nothing found
        return None

    if len(ret_cand_1) < 10:
        return ret_cand_1

    return ret_cand_1[:10]


def get_carrier_l(origin, dest):
    """
    Populates it w/ all carriers 

    """

    # get the three letter codes from origin, dest

    return IATA_AIRLINES_CODES_l + IATA_CODES_AIRLINES_l

    
def get_carrier_l_old(origin, dest):
    """
    gets the list from the params database

    """
    # get the three letter codes from origin, dest
    origin_upper, origin_valid = validate_airport(origin)
    dest_upper, dest_valid = validate_airport(dest)

    if origin_valid and dest_valid:  # return carriers

        exec_q = """
        SELECT DISTINCT(carrier) 
        FROM params 
        WHERE orig = '{0}' AND dest = '{1}'
        """.format(origin_upper, dest_upper)

        with MysqlConnectorEnv() as mconn:
            df1 = pd.read_sql_query(exec_q, mconn)

        ret_cand_1 = list(df1['carrier'])
        if len(ret_cand_1) == 0:  # no result
            return None

        # we have to return all the candidates
        # extend with flight names
        ret_cand_1.extend([iata_codes_airlines[x] for x in ret_cand_1])
        return ret_cand_1

    # wrong inputs
    return None
