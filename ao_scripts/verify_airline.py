# Verify origin airport

# AirOption modules
from ao_codes import iata_airlines_codes as IATA_AIRLINES_CODES,\
                     iata_codes_airlines as IATA_CODES_AIRLINES


def is_valid_airline(airline):
    """
    Returns true if the airline is valid, otherwise false
    returns the list of places from term

    :param airline: IATA code of the airline to be searched
    :type airline:  str
    :returns:       True if airline is valid, False if not
    :rtype:         bool
    """

    if airline is None:
        return False

    ret_cand_1 = filter( lambda x: x.upper().startswith(airline.upper())
                       , IATA_AIRLINES_CODES.keys() + IATA_CODES_AIRLINES.keys() )
    return not len(ret_cand_1) == 0  # TODO: PERHAPS THIS CAN BE RELAXED
