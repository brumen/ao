#
# processing of data obtained from webpage
#

import datetime
import logging

from typing          import Tuple
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


def validate_and_get_data(form) -> [Tuple, None]:
    """ Validates the data and constructs the input for the Air Option pricer.

    :param form: Input form from the webpage.
    :type form: ImmutableMultiDict (from Flask)
    :returns: A tuple of inputs for the Air Option pricer.
    """

    origin_place   = get_city_code(form.get('origin'))  # this also validates origin.
    dest_place     = get_city_code(form.get('dest'  ))  # validates destination as well.
    outbound_start = validate_dates(form.get('outbound_start'))
    outbound_end   = validate_dates(form.get('outbound_end'))

    # check that outbound_start < outbound_end
    if not outbound_start or not outbound_end or not (outbound_start <= outbound_end):
        return None

    carrier = get_airline_code(form.get('airline_name'))

    # getting the return flight information
    return_ow   = form.get('return_ow')  # this is either 'return' or 'one-way'
    cabin_class = form.get('cabin_class')  # cabin class
    nb_people   = form.get('nb_people')

    if not origin_place or not dest_place or not outbound_start or not outbound_end or not carrier or not strike:
        return None

    if return_ow == 'one-way':  # one-way flight
        option_start_ret, option_end_ret, inbound_start, inbound_end = None, None, None, None

    else:  # return flight
        option_start_ret = validate_dates(form.get('option_ret_start'  ))
        option_end_ret   = validate_dates(form.get('option_ret_end'    ))
        inbound_start    = validate_dates(form.get('outbound_start_ret'))
        inbound_end      = validate_dates(form.get('outbound_end_ret'  ))

        # check that inbound_start < inbound_end
        if not (inbound_start <= inbound_end) or not (outbound_end < inbound_start):
            return None

    # # TODO: CHECK FIRST OUTBOUND/INBOUND start
    return return_ow\
         , origin_place\
         , dest_place\
         , outbound_start\
         , outbound_end\
         , outbound_start\
         , outbound_end\
         , validate_strike(form.get('ticket_price'))\
         , None if carrier == '' else carrier\
         , option_start_ret\
         , option_end_ret\
         , inbound_start\
         , inbound_end\
         , cabin_class\
         , nb_people
