# Script for computing the air option

import datetime
import logging
import json

from typing          import Dict, Union, Any, Optional, Generator
from dateutil.parser import parse

from werkzeug.datastructures import ImmutableMultiDict

from ao.air_option_derive import AirOptionSkyScanner
from ao.iata.codes        import get_airline_code, get_city_code

# logger declaration
logger = logging.getLogger(__name__)


def validate_dates(date_ : str) -> Dict[str, Union[datetime.date, str]]:
    """ Validates whether date_ can be converted from format ('1/2/2017')
        to a datetime.date format and converts it. Otherwise return None.

    :param date_: date in a format that can be converted to / format ('1/2/2017')
    :returns: dictionary, where on success we return a dictionary w/
           key 'date', and value of type date in datetime.date format
           , otherwise we return a dict with key 'errors'
    """

    try:
        return {'date': parse(date_).date()}

    except TypeError as te:
        logger.error(f'{date_} in incorrect type: {str(te)}')
        return {'errors': f'{date_} in incorrect type: {str(te)}'}

    except ValueError as ve:
        logger.error(f'Could not convert {date_} to date format: {str(ve)}')
        return {'errors': f'Could not convert {date_} to date format: {str(ve)}'}

    except Exception as e:
        logger.error(f'Unknown exception: {e}')
        return {'errors': f'Unknown exception: {e}'}


def validate_strike(strike_i) -> Dict[str, Union[float, str]]:
    """ Tests whether strike_i is a float and returns appopriately

    :param strike_i: option strike, supposedly a float
    :returns: converted strike to float
              if conversion succeeded, return a dict w/ keyword 'strike', and the value is
                    the float of the strike
              if the conversion failed, return a dict w/ keyword 'errors' and a string which
                    reports the error.
    """

    try:
        return {'strike': float(strike_i)}

    except ValueError as ve:
        logger.error(f'Could not convert {strike_i} to float format: {str(ve)}')
        return {'errors': f'Could not convert {strike_i} to float format: {str(ve)}'}

    except TypeError as te:
        logger.error(f'Strike {strike_i} is not given in the floating point format: {str(te)}')
        return {'errors': f'Strike {strike_i} is not given in the floating point format: {str(te)}'}

    except Exception as e:
        logger.error(f'Unknown exception: {e}')
        return {'errors': f'Unknown exception: {e}'}


def validate_and_get_data(form : ImmutableMultiDict ) -> Union[Dict, None]:
    """ Validates the data and constructs the input for the Air Option pricer.

    :param form: Input form from the webpage.
    :returns: A tuple of inputs for the Air Option pricer.
    """

    # getting the return flight information
    return_ow   = form.get('return_ow')  # this is either 'return' or 'one-way'
    cabin_class = form.get('cabin_class')  # cabin class
    nb_people   = int(form.get('nb_people'))  # TODO: THIS HERE

    errors = []  # collection of errors if they occur

    strike      = validate_strike(form.get('ticket_price'))
    if 'errors' in strike:
        errors.append(strike['errors'])
    else:
        strike = strike['strike']

    # handling origin
    form_origin = form.get('origin')
    origin_place = get_city_code(form_origin)  # this also validates origin.
    if not origin_place:  # empty list
        errors.append(f'Could not find origin airport for {form_origin}')
    if len(origin_place) > 1:
        errors.append(f'Could not uniquely determine origin airport {origin_place}')
    origin_place = origin_place[0]  # has exactly 1 element

    # handling destination
    form_dest = form.get('dest')
    dest_place     = get_city_code(form_dest)  # validates destination as well.
    if not dest_place:  # empty list
        errors.append(f'Could not find destination airport for {form_dest}')
    if len(dest_place) > 1:
        errors.append(f'Could not uniquely determine dest airport from {origin_place}')
    dest_place = dest_place[0]  # has exactly 1 element

    # handling outbound start date
    outbound_start = validate_dates(form.get('outbound_start'))
    if 'errors' in outbound_start:
        errors.append(outbound_start['errors'])
    else:
        outbound_start = outbound_start['date']

    # handling outbound_end date
    outbound_end = validate_dates(form.get('outbound_end'))
    if 'errors' in outbound_end:
        errors.append(outbound_end['errors'])
    else:
        outbound_end = outbound_end['date']

    # checking the date condition
    if outbound_end <= outbound_start:
        errors.append(f'Outbound start date {outbound_start} > outbound end date {outbound_end}')

    # handling carrier
    airline_name = form.get('airline_name')
    carrier = get_airline_code(airline_name)
    if not carrier:  # no carriers
        errors.append(f'No carrier associated w/ {airline_name}')
    if len(carrier) > 1:
        errors.append(f'More than 1 airline associated /w {airline_name}.')
    carrier = carrier[0]

    if return_ow == 'one_way' and errors:
        return {'errors': errors}

    # one-way results
    result = { 'origin'             : origin_place
             , 'dest'               : dest_place
             , 'outbound_date_start': outbound_start
             , 'outbound_date_end'  : outbound_end
             , 'carrier'            : None if carrier == '' else carrier
             , 'cabinclass'         : cabin_class
             , 'adults'             : nb_people
             , 'K'                  : strike
             , }

    if return_ow == 'one_way':  # no errors
        return result

    # return-flight
    option_start_ret = validate_dates(form.get('option_ret_start'))
    if 'errors' in option_start_ret:
        errors.append(option_start_ret['errors'])
    else:
        option_start_ret = option_start_ret['date']

    option_end_ret   = validate_dates(form.get('option_ret_end'))
    if 'errors' in option_end_ret:
        errors.append(option_end_ret['errors'])
    else:
        option_end_ret = option_end_ret['date']

    inbound_start    = validate_dates(form.get('outbound_start_ret'))
    if 'errors' in inbound_start:
        errors.append(inbound_start['errors'])
    else:
        inbound_start = inbound_start['date']

    inbound_end      = validate_dates(form.get('outbound_end_ret'))
    if 'errors' in inbound_end:
        errors.append(inbound_end['errors'])
    else:
        inbound_end = inbound_end['errors']

    if outbound_end < inbound_start:
        errors.append(f'Outbound end {outbound_end} < inbound start {inbound_start}')

    if errors:  # errors for return flight
        return {'errors': errors}

    # no errors, return the results
    result.update({ 'inbound_date_start'   : inbound_start
                  , 'inbound_date_end'     : inbound_end
                  , 'return_flight'        : True
                  , })
    #'option_ret_start_date': option_start_ret
    #, 'option_ret_end_date': option_end_ret
    #})

    return result


def compute_option( form          : ImmutableMultiDict
                  , recompute_ind : bool = False
                  , market_date   : Optional[datetime.date] = None ) -> Generator[Dict, None, None]:
    """ Interface to the compute_option_val function from air_option, to
        be called from the flask interface

    :param form: "Dict" of parameters passed from the browser
    :param recompute_ind: indicator whether to do a recomputation or not
    :param market_date: optional market date, otherwise use today as market date.
    :returns: generator with keys 'finished': true or false,
                  'result': if finished is False, report on data acquisition, if true, report result.
    """

    validated_data = validate_and_get_data(form)

    # TODO: The recompute part.
    if recompute_ind:  # recompute part
        sel_flights_dict = json.loads(form['flights_selected'])  # in dict form

    # Initiating data transfer
    if 'errors' in validated_data:  # validated_data, validation has errors
        yield {'finished': True, 'error': validated_data['errors']}
        return  # stopping the iteration after error ecountered.

    date_used = datetime.date.today() if market_date is None else market_date

    # data are OK, construct a skyscanner object
    ao = AirOptionSkyScanner(date_used, **validated_data)

    # handle flight acquisition
    for flight in ao.flights:  # this is a generator, get flights
        yield {'finished': False, 'result': f'Got flight {flight}'}

    # produce the PV result after all the flights are obtained.
    yield {'finished': True, 'result': ao.PV()}
