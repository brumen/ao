# Module for checking that the origin is valid

from typing import List

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

    ret_cand_1 = list(filter( lambda x: airport_partial_name.upper() in x
                            , IATA_CITIES_CODES_l + IATA_CODES_CITIES_l ) )

    if not ret_cand_1:  # nothing found, report problem
        return []             , False

    elif len(ret_cand_1) < 10:
        return ret_cand_1     , True

    else:
        return ret_cand_1[:10], True


def show_airline_l(airline_partial_name : str) -> tuple:
    """
    Returns the list of IATA airlines for the partial name searched.

    :param airline_partial_name: partial name of the airline one is searching for.
    :returns: list of names one for the airlines one is searching for.
    """

    ret_cand_1 = list(filter( lambda x: airline_partial_name.upper() in x.upper()
                            , IATA_AIRLINES_CODES_l) )

    if not ret_cand_1:  # nothing found
        return []             , False

    if len(ret_cand_1) < 10:
        return ret_cand_1     , True
    else:
        return ret_cand_1[:10], True
