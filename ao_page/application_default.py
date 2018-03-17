# Handling the requests of the webpage
import logging
import json
import os.path
import datetime

import config
import ao_codes

from time             import localtime
from threading        import Thread
from sse              import Publisher

from logging.handlers import MemoryHandler
from flask            import Flask, request, jsonify, Response

from ao_scripts.find_relevant_carriers import get_carrier_l
from ao_scripts.verify_origin          import is_valid_origin \
                                            , is_valid_airline \
                                            , show_airline_l \
                                            , show_airport_l
from ao_scripts.compute_option         import compute_option


logger_format = '%(asctime)s:%(name)s:%(levelname)s:%(message)s'


class AOParsingFilter(logging.Filter):
    """
    Filters out the records which AO produces
    """

    def filter(self, record):
        return not record.getMessage().startswith('AO')


# logger setup
logging.basicConfig( filename = os.path.join(config.log_dir, 'ao.log')
                   , level    = logging.CRITICAL)
logger = logging.getLogger()  # root logger
logger.setLevel(logging.INFO)
# logger_handler = logging.FileHandler())
# logger_handler.setFormatter(logging.Formatter(logger_format))
# logger.addHandler(logger_handler)
# computeStream = ComputeStream()  # object keeping the messages
# stream_handler = logging.handlers.MemoryHandler( 0  # capcity = 0, immediately flush
#                                              , flushLevel = logging.INFO
#                                               , target     = computeStream )
# logger.addHandler(stream_handler)


# Flask app
app = Flask(__name__)
app.debug = True
app.use_debugger = False
# app.use_reloader = False

# publisher
publisher_ao = Publisher()


def time_now():
    """
    Returns local time in the string format.

    :returns: local time separated by underscores.
    :rtype: str
    """
    lt = localtime()
    return '_'.join([ str(lt.tm_year), str(lt.tm_mon), str(lt.tm_mday)
                    , str(lt.tm_hour), str(lt.tm_min), str(lt.tm_sec)])


@app.route('/verify_airline', methods = ['GET'])
def verify_airline():
    """
    Checks that the airline is correct
    """

    return jsonify({'found': is_valid_airline(request.args.get('airline', ''))})


@app.route('/verify_origin', methods = ['GET'])
def verify_origin():
    """
    Checks whether the airport is a valid IATA name

    """

    return jsonify ({'found': is_valid_origin(request.args.get('origin', ''))})


@app.route('/find_relevant_carriers', methods = ['GET'])
def find_relevant_carriers():

    origin = request.args.get('origin', '')
    dest   = request.args.get('dest'  , '')
    is_valid, return_l = get_carrier_l(origin, dest)

    return jsonify({ 'is_valid'     : is_valid
                   , 'list_carriers': return_l})


@app.route('/recompute_option', methods = ['POST'])
def recompute_option_flask():
    """
    Recomputes the option value,
    makes the same call as the compute_option with the additional flag.
    No need for publisher here, as this is fast.

    """

    # compute_id identifies the computation request
    return jsonify(compute_option( request.get_json()
                                 , recompute_ind = True
                                 , compute_id    = str(datetime.datetime.now()) ) )


@app.route('/write_inquiry', methods=['POST'])
def write_inquiry():
    """
    Writes a file about the inquiry to the inquiry folder.

    """

    with open(ao_codes.inquiry_dir + '/inquiry_solo/' +
              'inquiry_' + time_now() + '.inq', 'w') as fo:
        fo.write(json.dumps(request.get_json()))

    # succeeds, returns true
    return jsonify({'valid': True})


@app.route('/compute_option', methods = ['GET'])
def compute_option_flask():
    """
    Computes the option w/ server sent events (SSE)

    """

    publisher_ao_local = Publisher()

    Thread( target = compute_option
          , args   = (request.args, )
          , kwargs = { "publisher_ao" : publisher_ao_local  # publisher_ao
                     , "recompute_ind": False
                     , 'compute_id': str(datetime.datetime.now()) } ).start()

    return Response( publisher_ao_local.subscribe()
                   , mimetype="text/event-stream" )


@app.route('/ao_auto_fill_origin')
def ao_auto_fill_origin():
    """
    Returns the auto fill of the IATA origin airports

    """

    airport_l, found_ind = show_airport_l(request.args.get("term"))
    return Response( json.dumps(airport_l)  # jsonify doesnt work
                   , mimetype = 'application/json')


@app.route('/ao_auto_fill_airline')
def ao_auto_fill_airline():
    """
    Returns the auto fill of the IATA airline codes

    """

    airline_l, found_ind = show_airline_l(request.args.get("term"))
    return Response( json.dumps(airline_l)
                   , mimetype = 'application/json' )

#@app.route('/ao_payment', methods = ['GET'])
#def ao_payment():
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
