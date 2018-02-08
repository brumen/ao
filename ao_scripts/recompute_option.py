# re-computes the option from the forwards given
import logging
import numpy as np
import json

# ao local modules
from air_option import compute_option_val
from get_data   import get_data_dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def recompute_option(form):
    """

    """

    data_very_raw = get_data_dict(form)  # TODO: THIS NEEDS TO CHANGE

    is_one_way = len(data_very_raw) == 12  # is the flight return or one-way

    if is_one_way:  # one-way flight
        all_valid, origin_place, dest_place, option_start, option_end, \
            outbound_start, outbound_end, strike, carrier_used, return_ow, \
            cabin_class, nb_people = data_very_raw
    else:  # return flight, more to unpack
        all_valid, origin_place, dest_place, option_start, option_end, \
            outbound_start, outbound_end, strike, carrier_used, \
            option_start_ret, option_end_ret, inbound_start, inbound_end, \
            return_ow, cabin_class, nb_people = data_very_raw

    # flights to select
    sel_flights = form['flights_selected']  # in string form, convert to dict
    sel_flights_dict = json.loads(sel_flights)  # in dict form

    if not all_valid:
        return False, {}

    else:

        way_args = { 'origin_place': origin_place
                   , 'dest_place':   dest_place
                   , 'flights_include': sel_flights_dict
                   , 'option_start_date': option_start
                   , 'option_end_date': option_end
                   , 'outbound_date_start': outbound_start
                   , 'outbound_date_end': outbound_end
                   , 'carrier': carrier_used
                   , 'cabinclass': cabin_class
                   , 'adults' :np.int(nb_people)
                   , 'K':  np.double(strike)
                   , 'recompute_ind': True }

        if not is_one_way:
            way_args.update({ 'option_ret_start_date': option_start_ret
                            , 'option_ret_end_date': option_end_ret
                            , 'inbound_date_start': inbound_start
                            , 'inbound_date_end': inbound_end
                            , 'return_flight': True})

        # do the recomputation
        result, price_range, flights_v, reorg_flights_v, minmax_v = \
            compute_option_val(way_args)

        if result == 'Invalid':
            logger.info('Invalid results of recomputation')

            return False, {}  # invalid entries

        else:
            logger.info('Recomputation finished')

            return True, {'valid_inp': True,
                          'return_ind': not is_one_way,
                          'price': result,
                          'flights': flights_v,
                          'reorg_flights': reorg_flights_v,
                          'minmax': minmax_v,
                          'price_range': price_range,
                          'do_nothing': True }
