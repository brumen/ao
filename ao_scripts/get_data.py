# getting data from the webpage
import datetime
from dateutil.parser import parse

# ao modules 
import ds
from ao_codes import iata_cities_codes, iata_codes_cities, \
                     iata_airlines_codes, iata_codes_airlines


def validate_airport(airport):
    """
    extract the code from the airport if code not given
    1. if code given, return code, 
    2. if airport name given, look into db 

    :param airport: IATA code or the full airport name
    :type airport:  str
    :returns:       tuple of (IATA name, True/False if the airport was found)
    :rtype:         tuple of (str, bool)
    """

    airport_upper = airport.upper()

    if airport_upper in iata_codes_cities.keys():  # airport is in IATA code give
        return airport_upper, True

    else:  # airport has a name
        airport_keys_upper = [x.upper() for x in iata_cities_codes.keys()]

        if airport_upper in airport_keys_upper:
            airport_idx  = airport_keys_upper.index(airport_upper)
            airport_name = iata_cities_codes.keys()[airport_idx]
            return iata_cities_codes[airport_name], True  # return the code

        else:
            return 'Invalid', False


def validate_airline(airline):
    """
    extract the code from the airline if code not given
    1. if code given, return code,
    2. if airline name given, look into db

    :param airline: IATA code or the full airport name
    :type airline:  str
    :returns:       tuple of (IATA name, True/False if the airport was found)
    :rtype:         tuple of (str, bool)
    """

    airline_upper = airline.upper()

    if airline_upper in iata_codes_airlines.keys():
        return airline_upper, True

    else:
        iata_airlines_upper = [x.upper() for x in iata_airlines_codes.keys()]

        if airline_upper in iata_airlines_upper:
            airline_idx = iata_airlines_upper.index(airline_upper)
            airline_name = iata_airlines_codes.keys()[airline_idx]
            return iata_airlines_codes[airline_name], True
        else:
            return 'Invalid', False


def validate_dates(date_):
    """
    Validates whether date_ is in the 1/2/2017 format, and converts it into - format

    :param date_: date in / format ('1/2/2017')
    :type date: str
    :returns: date in - format '2017-01-05'
    :rtype: str
    """

    try:
        date_dt = parse(date_).date()
    except Exception:
        return '', False

    return date_dt, True  # if everything OK, then return


def validate_strike(strike_i):
    """
    Tests whether strike_i is a float and returns appopriately

    :param strike_i: option strike, supposedly a float
    :type strike_i: possibly float
    :returns: converted strike to float, True/False whether the conversion succeeded
    :rtype: tuple (float, bool)
    """

    try:
        strike_i_float = float(strike_i)
    except ValueError:
        return -1., False

    return strike_i_float, True


def get_data(form):
    """
    obtains data from the form and returns them 

    :param form:
    :type form: ImmutableMultiDict (from Flask)
    :returns:
    :rtype:
    """

    all_valid = True
    origin_place,   origin_valid = validate_airport(form.get('origin_place'))
    dest_place,     dest_valid   = validate_airport(form.get('dest_place'))
    outbound_start, obs_valid    = validate_dates(form.get('outbound_start'))
    outbound_end,   obe_valid    = validate_dates(form.get('outbound_end'))
    option_start,   os_valid     = outbound_start, True  # THIS IS WRONG
    option_end,     oe_valid     = outbound_end, True

    # check that outbound_start < outbound_end
    if obs_valid and obe_valid:
       all_valid         = all_valid and (outbound_start <= outbound_end)

    strike,  strike_valid  = validate_strike(form.get('ticket_price'))
    carrier, carrier_valid = validate_airline(form.get('airline_name'))

    # getting the return flight information
    return_ow   = form.get('return_ow')  # this is either 'return' or 'one-way'
    cabin_class = form.get('cabin_class')  # cabin class
    nb_people   = form.get('nb_people')
    all_valid   = all_valid and origin_valid and dest_valid and obs_valid and obe_valid and\
                  carrier_valid and strike_valid

    if return_ow == 'one-way':  # one-way flight
        option_start_ret, option_end_ret, inbound_start, inbound_end = None, None, None, None
    else:  # return flight
        option_start_ret, ors_valid = validate_dates(form.get('option_ret_start'))
        option_end_ret,   ore_valid = validate_dates(form.get('option_ret_end'))
        inbound_start,    ibs_valid = validate_dates(form.get('outbound_start_ret'))
        inbound_end,      ibe_valid = validate_dates(form.get('outbound_end_ret'))

        # check that inbound_start < inbound_end
        if ibs_valid and ibe_valid:
            all_valid        = all_valid and (inbound_start <= inbound_end) and \
                               (outbound_end < inbound_start)

        all_valid = all_valid and ibs_valid and ibe_valid
        
    if carrier == '':
        carrier_used = None
    else:
        carrier_used = carrier

    return return_ow, ( all_valid
                      , origin_place
                      , dest_place
                      , option_start
                      , option_end
                      , outbound_start
                      , outbound_end
                      , strike
                      , carrier_used
                      , option_start_ret
                      , option_end_ret
                      , inbound_start
                      , inbound_end
                      , return_ow
                      , cabin_class
                      , nb_people )


def get_data_final(form):
    """
    obtains data from the form (from form _final for booking) and returns them

    """

    # TODO: check if email is a real address
    return_ow, results = get_data(form)

    return return_ow, results, form.get('email-addr')
