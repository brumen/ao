# Handling the requests of the webpage
import logging
import json
import uuid
import unirest
import os.path

import config
import ao_codes

from time             import sleep, localtime
from sse              import Publisher



from logging.handlers import MemoryHandler
from flask            import Flask, request, jsonify, Response

from ao_scripts.find_relevant_carriers import get_carrier_l
from ao_scripts.verify_airline         import is_valid_airline
from ao_scripts.verify_origin          import is_valid_origin
from ao_scripts.compute_option         import compute_option
from ao_scripts.ao_auto_fill_origin    import show_airline_l, show_airport_l


loggger_format = '%(asctime)s:%(name)s:%(levelname)s:%(message)s'


class AOParsingFilter(logging.Filter):
    """
    Filters out the records which AO produces
    """

    def filter(self, record):
        return not record.getMessage().startswith('AO')


class ComputeStream(object):

    def __init__(self):
        self.__allMessages = []

    def handle(self, record):
        """
        Handles the record, appends it to __allMessages

        """
        self.__allMessages.append(record)

    def __filterRecord(self, record):
        """
        Returns true if record satisfies certain conditions

        """

        recordList = record.split(":")
        recordList[3:]  # everything after: asctime, name, levelname
        return record

    def __processMessage(self, record):
        return record

    def getMessages(self):
        """
        Retrieve all the messages in the form of a generator.

        """

        while True:
            sleep(1)  # sleep 1 second

            while self.__allMessages:  # while this is not empty

                currMessage = self.__allMessages.pop(0)

                if self.__filterRecord(currMessage):
                    yield "data: {0}\n\n".format(self.__processMessage(currMessage))
                # otherwise discard the message, not relevant for printing


# logger setup
logger = logging.getLogger()  # root logger
logger.setLevel(logging.INFO)
logger_handler = logging.FileHandler(os.path.join(config.log_dir, 'ao.log'))
logger_handler.setFormatter(logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s'))
logger.addHandler(logger_handler)
computeStream = ComputeStream()  # object keeping the messages
stream_handler = logging.handlers.MemoryHandler( 0  # capcity = 0, immediately flush
                                               , flushLevel = logging.INFO
                                               , target     = computeStream )
logger.addHandler(stream_handler)

# Flask app
app = Flask(__name__)
app.debug = True
app.use_debugger = False
# app.use_reloader = False


def time_now():
    """
    Returns local time in the string format.

    :returns: local time separated by underscores.
    :rtype: str
    """
    lt = localtime()
    return '_'.join([ str(lt.tm_year), str(lt.tm_mon), str(lt.tm_mday)
                    , str(lt.tm_hour), str(lt.tm_min), str(lt.tm_sec)])


# TODO: REMOVE THIS IN THE FINAL APPLICATION
@app.route('/')
def hello_world():
    return 'Hello, World!'


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
    Recomputes the option value, makes the same call as the compute_option with the additional flag.

    """

    return compute_option( request.args
                         , recompute_ind = True)


@app.route('/write_inquiry', methods=['POST'])
def write_inquiry():
    """
    Writes a file about the inquiry to the inquiry folder

    """

    with open(ao_codes.inquiry_dir + '/inquiry_solo/' +
              'inquiry_' + time_now() + '.inq', 'w') as fo:
        # TODO: THIS NEEDS FIXING
        fo.write(json.dumps(request.form))

    # succeeds, returns true
    return jsonify({'valid': True})


@app.route('/compute_option', methods = ['GET'])
def compute_option_flask():
    """
    Computes the option w/ server sent events (SSE)

    """

    publisher_ao = Publisher()
    res = compute_option(request.args
                        , publisher_ao  = publisher_ao
                        , recompute_ind = False )

    return Response( publisher_ao.subscribe()  # this will be returning the messages
                   , mimetype="text/event-stream" )


@app.route('/ao_auto_fill_origin')
def ao_auto_fill_origin():
    """
    Returns the auto fill of the IATA origin airports


    """

    # TODO: WHERE DO WE NEED found_ind

    return_l, found_ind = show_airport_l(request.args.get("term"))
    return jsonify(return_l)


@app.route('/ao_auto_fill_airline')
def ao_auto_fill_airline():
    """
    Returns the auto fill of the IATA airline codes

    """

    # TODO: WHERE DO WE NEED found_ind
    return_l, found_ind = show_airline_l(request.args.get("term"))
    return jsonify(return_l)


# TODO: FIX THIS ROUTE HERE
@app.route('/ao_payment', methods = ['GET'])
def ao_payment():
    # The following variables need to be assigned:
    #   card_nonce
    #   location_id
    #   access_token

    card_nonce = request.args.get('card_nonce', '')
    location_id = request.args.get('location_id', '')
    access_token = request.args.get('access_token', '')

    # work that is done
    response = unirest.post( 'https://connect.squareup.com/v2/locations/' + location_id + '/transactions'
                           , headers = { 'Accept': 'application/json'
                                       , 'Content-Type': 'application/json'
                                       , 'Authorization': 'Bearer ' + access_token
                                       ,}
                           , params = json.dumps({ 'card_nonce': card_nonce
                                                 , 'amount_money': { 'amount': 100
                                                                   , 'currency': 'USD' }
                                                 , 'idempotency_key': str(uuid.uuid1()) }) )

    return response.body
