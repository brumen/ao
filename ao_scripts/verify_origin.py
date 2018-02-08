# Module for checking that the origin is valid

from ao_codes import iata_cities_codes as IATA_CITIES_CODES,\
                     iata_codes_cities as IATA_CODES_CITIES


def is_valid_origin(origin):
    """
    Returns true if origin airport is in a list of airports

    :param origin: IATA name, or full name of the origin airport
    :type origin:  str
    :returns:      True if origin is in the list of airports
    :rtype:        bool
    """

    if origin is None:
        return False

    ret_cand_1 = filter( lambda x: x.startswith(origin.upper())
                       , IATA_CITIES_CODES.keys() + IATA_CODES_CITIES.keys() )

    return not len(ret_cand_1) == 0
