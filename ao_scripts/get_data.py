#
# getting data from the webpage
#

import datetime
import logging

from dateutil.parser import parse

from iata.codes import get_airline_code, get_city_code

logger = logging.getLogger(__name__)


def validate_dates(date_ : str) -> datetime.date:
    """
    Validates whether date_ can be converted from format ('1/2/2017')
    to a datetime.date format and converts it. Otherwise return None.

    :param date_: date in a format that can be converted to / format ('1/2/2017')
    :returns: date in datetime.date format, or None if not given in any form.
    """

    try:
        return parse(date_).date()
    except Exception:
        return None


def validate_strike(strike_i) -> float:
    """
    Tests whether strike_i is a float and returns appopriately

    :param strike_i: option strike, supposedly a float
    :type strike_i: possibly float
    :returns: converted strike to float, or None if the conversion did _not_ succeed
    """

    try:
        return float(strike_i)
    except ValueError:
        return None


# TODO: REWRITE THIS PART HERE!!!
def get_data(form):
    """ Obtains data from the form and returns them.

    :param form:
    :type form: ImmutableMultiDict (from Flask)
    :returns:
    :rtype:
    """

    origin_place   = get_city_code(form.get('origin_place'))
    dest_place     = get_city_code(form.get('dest_place'  ))
    outbound_start = validate_dates(form.get('outbound_start'))
    outbound_end   = validate_dates(form.get('outbound_end'))

    # check that outbound_start < outbound_end
    all_valid = True
    if outbound_start and outbound_end:
       all_valid = outbound_start <= outbound_end

    carrier = get_airline_code(form.get('airline_name'))

    # getting the return flight information
    return_ow   = form.get('return_ow')  # this is either 'return' or 'one-way'
    cabin_class = form.get('cabin_class')  # cabin class
    nb_people   = form.get('nb_people')
    all_valid   = all_valid and origin_place and dest_place and outbound_start and outbound_end and carrier and strike

    if return_ow == 'one-way':  # one-way flight
        option_start_ret, option_end_ret, inbound_start, inbound_end = None, None, None, None

    else:  # return flight
        option_start_ret = validate_dates(form.get('option_ret_start'  ))
        option_end_ret   = validate_dates(form.get('option_ret_end'    ))
        inbound_start    = validate_dates(form.get('outbound_start_ret'))
        inbound_end      = validate_dates(form.get('outbound_end_ret'  ))

        # check that inbound_start < inbound_end
        if inbound_start and inbound_end:
            all_valid = all_valid and (inbound_start <= inbound_end) and (outbound_end < inbound_start)


    return return_ow, ( all_valid
                      , origin_place
                      , dest_place
                      , outbound_start # TODO: CHECK THIS
                      , outbound_end   # TODO: CHECK THIS
                      , outbound_start
                      , outbound_end
                      , validate_strike(form.get('ticket_price'))
                      , None if carrier == '' else carrier
                      , option_start_ret
                      , option_end_ret
                      , inbound_start
                      , inbound_end
                      , return_ow
                      , cabin_class
                      , nb_people )


def get_data_final(form):
    """ Obtains data from the form (from form _final for booking) and returns them

    """

    # TODO: check if email is a real address
    return_ow, results = get_data(form)

    return return_ow, results, form.get('email-addr')
