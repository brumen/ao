# Auto fill airport destination

from ao_codes import iata_cities_codes, iata_codes_cities
from ao_codes import iata_airlines_codes as IATA_AIRLINES_CODES


def show_airport_l(term):
    """
    returns the list of places from term 

    """

    ret_cand_1 = filter( lambda x: term.upper() in x
                       , list(iata_cities_codes.keys()) + list(iata_codes_cities.keys()) )

    if not ret_cand_1:  # nothing found, report problem
        return []             , False

    elif len(ret_cand_1) < 10:
        return ret_cand_1     , True

    else:
        return ret_cand_1[:10], True


def show_airline_l(term):
    """
    returns the list of places from term

    """

    ret_cand_1 = filter( lambda x: term.upper() in x.upper()
                       , list(IATA_AIRLINES_CODES.keys()))

    if len(ret_cand_1) < 10:
        return ret_cand_1
    else:
        return ret_cand_1[:10]
