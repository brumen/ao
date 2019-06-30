# Handling the requests of the webpage
import logging
import json
import os.path
import datetime

import config

from time  import localtime
from flask import Flask, request, jsonify, Response

from ao_codes                  import inquiry_dir
from iata.codes                import get_airline_code, get_city_code
from ao_scripts.verify_origin  import get_carriers_on_route
from ao_scripts.compute_option import compute_option


logging.basicConfig( filename = os.path.join(config.log_dir, 'ao.log')
                   , level    = logging.CRITICAL)

logger = logging.getLogger()

# Flask app
ao_rester = Flask(__name__)
ao_rester.debug = True
ao_rester.use_debugger = False


def time_now():
    """
    Returns local time in the string format.

    :returns: local time separated by underscores.
    :rtype: str
    """

    lt = localtime()
    return '_'.join([ str(lt.tm_year), str(lt.tm_mon), str(lt.tm_mday)
                    , str(lt.tm_hour), str(lt.tm_min), str(lt.tm_sec)])


@ao_rester.route('/verify_airline', methods = ['GET'])
def verify_airline():
    """
    Checks that the airline is correct
    """

    # TODO:  THIS CAN RETURN MULTIPLE AIRLINES - CHECK IF THIS WORK
    airline_code = get_airline_code(request.args.get('airline', ''))

    return jsonify({'found': airline_code})


@ao_rester.route('/verify_origin', methods = ['GET'])
def verify_origin():
    """
    Checks whether the airport is a valid IATA name

    """

    # TODO: THIS CAN RETURN MULTIPLE CITIES
    city_code = get_city_code(request.args.get('origin', ''))
    return jsonify ({'found': city_code})


@ao_rester.route('/find_relevant_carriers', methods = ['GET'])
def find_relevant_carriers():
    """ Endpoint for finding the relevant carriers from origin to dest.

    """

    carriers = get_carriers_on_route( request.args.get('origin', '')
                                    , request.args.get('dest'  , '') )

    return jsonify({ 'is_valid'     : carriers is not None
                   , 'list_carriers': carriers})


@ao_rester.route('/recompute_option', methods = ['POST'])
def recompute_option_flask():
    """
    Recomputes the option value,
    makes the same call as the compute_option with the additional flag.
    No need for publisher here, as this is fast.

    """

    return jsonify(compute_option( request.get_json()
                                 , recompute_ind = True ) )


@ao_rester.route('/write_inquiry', methods=['POST'])
def write_inquiry():
    """ Writes a file about the inquiry to the inquiry folder.

    """

    with open(inquiry_dir + '/inquiry_solo/' + 'inquiry_' + time_now() + '.inq', 'w') as fo:
        fo.write(json.dumps(request.get_json()))

    # succeeds, returns true
    return jsonify({'valid': True})


# TODO: CHECK IF THIS IS GET???/
@ao_rester.route('/compute_option', methods = ['GET'])
def compute_option_flask():
    """ Computes the option w/ server sent events (SSE)

    """

    # compute_option has to be a generator

    return Response(compute_option(request.args), mimetype="text/event-stream")


@ao_rester.route('/compute_option_now', methods = ['GET'])
def compute_option_now_flask():
    """ Immediate response compute option.

    """

    return jsonify(compute_option(request.get_json()))


@ao_rester.route('/ao_auto_fill_origin')
def ao_auto_fill_origin():
    """
    Returns the auto fill of the IATA origin airports

    """

    airports = get_city_code(request.args.get("term"))
    return Response( json.dumps(airports)  # jsonify doesnt work
                   , mimetype = 'application/json')


@ao_rester.route('/ao_auto_fill_airline')
def ao_auto_fill_airline():
    """
    Returns the auto fill of the IATA airline codes

    """

    airline_l = get_airline_code(request.args.get("term"))
    return Response( json.dumps(airline_l)
                   , mimetype = 'application/json' )


# @ao_rester.route('/ao_payment', methods = ['GET'])
# def ao_payment():
#    # The following variables need to be assigned:
#    #   card_nonce
#    #   location_id
#    #   access_token

#    card_nonce = request.args.get('card_nonce', '')
#    location_id = request.args.get('location_id', '')
#    access_token = request.args.get('access_token', '')

#    # work that is done
#    response = unirest.post( 'https://connect.squareup.com/v2/locations/' + location_id + '/transactions'
#                           , headers = { 'Accept': 'application/json'
#                                       , 'Content-Type': 'application/json'
#                                       , 'Authorization': 'Bearer ' + access_token
#                                       ,}
#                           , params = json.dumps({ 'card_nonce': card_nonce
#                                                 , 'amount_money': { 'amount': 100
#                                                                   , 'currency': 'USD' }
#                                                 , 'idempotency_key': str(uuid.uuid1()) }) )
#
#    return response.body

# def __main__():
ao_rester.run()
