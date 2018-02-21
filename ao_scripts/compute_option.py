# Script for computing the air option

import numpy as np
import logging
import json

# ao modules
from air_option          import compute_option_val, data_yield
from ao_scripts.get_data import get_data

# logger declaration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def compute_option( form
                  , publisher_ao  = None
                  , recompute_ind = False ):
    """
    Interface to the compute_option_val function from air_option, to
    be called from the flask interface

    :param form: "Dict" of parameters passed from the browser
    :type form: ImmutableMultiDict
    :param publisher_ao: publisher object used for publishing the messages
    :type publisher_ao: sse.Publisher
    :param recompute_ind: indicator whether to do a recomputation or not
    :type recompute_ind: bool
    :returns:
    :rtype: dict
    """

    is_one_way, ( all_valid, origin_place, dest_place, option_start, option_end,
                  outbound_start, outbound_end, strike, carrier_used,
                  option_start_ret, option_end_ret, inbound_start, inbound_end,
                  return_ow, cabin_class, nb_people ) = get_data(form)

    if recompute_ind:  # recompute part
        sel_flights_dict = json.loads(form['flights_selected'])  # in dict form

    logger.info(';'.join(['AO', 'Initiating flight fetch.']))

    if publisher_ao:
        publisher_ao.publish(data_yield({ 'finished': False
                                        , 'result'  : 'Initiating flight fetch.' } ) )

    if not all_valid:  # dont compute, inputs are wrong
        logger.info(';'.join(['AO', 'Invalid input data.']))
        result_not_valid = { 'finished': True
                           , 'result': { 'finished': True
                                       , 'progress_notice': 'finished'  # also irrelevant
                                       , 'valid_inp'      : False } }
        if publisher_ao:
            publisher_ao.publish(data_yield( result_not_valid ) )

        if recompute_ind:  # recompute returns
            return result_not_valid

    else:
        way_args = { 'origin_place':        origin_place
                   , 'dest_place':          dest_place
                   , 'option_start_date':   option_start
                   , 'option_end_date':     option_end
                   , 'outbound_date_start': outbound_start
                   , 'outbound_date_end':   outbound_end
                   , 'carrier':             carrier_used
                   , 'cabinclass':          cabin_class
                   , 'adults':              nb_people
                   , 'K':                   np.double(strike)
                   , 'errors':              'ignore' }

        if return_ow != 'one-way':
            way_args.update( { 'option_ret_start_date': option_start_ret
                             , 'option_ret_end_date'  : option_end_ret
                             , 'inbound_date_start'   : inbound_start
                             , 'inbound_date_end'     : inbound_end
                             , 'return_flight'        : True } )

        if publisher_ao:
            way_args.update( { 'publisher_ao': publisher_ao } )

        if recompute_ind:
            way_args.update({ 'flights_include': sel_flights_dict
                            , 'recompute_ind'  : True })

        logger.info(';'.join(['AO', 'Starting option computation.']))

        result, price_range, flights_v, reorg_flights_v, minmax_v = \
            compute_option_val(**way_args)

        logger.info(';'.join(['AO', 'Finished option computation']))

        if result == 'Invalid':
            logger.info(';'.join(['AO', json.dumps((False, {}))]))
            if publisher_ao:
                publisher_ao.publish(data_yield({ 'finished': True
                                                , 'results' : {} }))

            if recompute_ind:  # recompute switch returns
                return { 'finished': True
                       , 'results' : {} }

        else:  # actual display
            final_result = {'finished': True
                           , 'result' : { 'finished': True
                                        , 'progress_notice': 'finished'  # also irrelevant
                                        , 'valid_inp'      : True
                                        , 'return_ind'     : return_ow
                                        , 'price'          : result
                                        , 'flights'        : flights_v
                                        , 'reorg_flights'  : reorg_flights_v
                                        , 'minmax'         : minmax_v
                                        , 'price_range'    : price_range} }

            logger.info(';'.join(['AO', data_yield(final_result)]))

            if publisher_ao:
                publisher_ao.publish(data_yield(final_result))

            if recompute_ind:  # recompute switch returns
                return final_result
