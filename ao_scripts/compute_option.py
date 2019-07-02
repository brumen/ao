# Script for computing the air option

import datetime
import logging
import json

from typing          import Tuple
from dateutil.parser import parse

from werkzeug.datastructures import ImmutableMultiDict

from air_option import AirOptionMock
from iata.codes import get_airline_code, get_city_code

# logger declaration
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


def validate_and_get_data(form : ImmutableMultiDict ) -> [Tuple, None]:
    """ Validates the data and constructs the input for the Air Option pricer.

    :param form: Input form from the webpage.
    :returns: A tuple of inputs for the Air Option pricer.
    """

    # getting the return flight information
    return_ow   = form.get('return_ow')  # this is either 'return' or 'one-way'
    cabin_class = form.get('cabin_class')  # cabin class
    nb_people   = form.get('nb_people')
    strike      = validate_strike(form.get('ticket_price'))

    # departure information
    origin_place   = get_city_code(form.get('origin'))  # this also validates origin.
    dest_place     = get_city_code(form.get('dest'  ))  # validates destination as well.
    outbound_start = validate_dates(form.get('outbound_start'))
    outbound_end   = validate_dates(form.get('outbound_end'))
    option_start   = validate_dates(form.get('option_start'))
    option_end     = validate_dates(form.get('option_end'))

    # check that outbound_start < outbound_end
    if not outbound_start or not outbound_end or not (outbound_start <= outbound_end):
        return None

    carrier = get_airline_code(form.get('airline_name'))

    if not origin_place or not dest_place or not outbound_start or not outbound_end or not carrier or not strike:
        return None

    # output gathering

    result = { 'origin_place': origin_place
             , 'dest_place': dest_place
             , 'option_start_date': option_start
             , 'option_end_date': option_end
             , 'outbound_date_start': outbound_start
             , 'outbound_date_end': outbound_end
             , 'carrier': None if carrier == '' else carrier
             , 'cabinclass': cabin_class
             , 'adults': nb_people
             , 'K': strike }

    if return_ow == 'one_way':
        return result

    # return-flight
    option_start_ret = validate_dates(form.get('option_ret_start'))
    option_end_ret = validate_dates(form.get('option_ret_end'))
    inbound_start = validate_dates(form.get('outbound_start_ret'))
    inbound_end = validate_dates(form.get('outbound_end_ret'))

    if not (inbound_start <= inbound_end) or not (outbound_end < inbound_start):
        return None

    result.update({ 'option_ret_start_date': option_start_ret
                  , 'option_ret_end_date': option_end_ret
                  , 'inbound_date_start': inbound_start
                  , 'inbound_date_end': inbound_end
                  , 'return_flight': True})

    return result


def compute_option( form
                  , recompute_ind = False ) -> dict:
    """
    Interface to the compute_option_val function from air_option, to
    be called from the flask interface

    :param form: "Dict" of parameters passed from the browser
    :type form: ImmutableMultiDict
    :param recompute_ind: indicator whether to do a recomputation or not
    :returns:
    """

    validated_data = validate_and_get_data(form)

    if recompute_ind:  # recompute part
        sel_flights_dict = json.loads(form['flights_selected'])  # in dict form

    # Initiating data transfer
    if not validated_data:  # validated_data = None
        yield { 'finished': True
              , 'result'  : None }
    else:
        yield { 'finished': False
              , 'result'  : 'Initiating flight fetch.' }

        # TODO: Handle recompute stuff later!!!
        ao = AirOptionMock(datetime.date.today(), **validated_data)
        for flight in ao.get_flights():  # this is a generator
            yield { 'finished': False
                  , 'result'  : 'Fetchingn flight {0}'.format(flight) }  # TODO: This flight here is WRONG !!!!

        yield { 'finished': True
              , 'result'  : ao() }
