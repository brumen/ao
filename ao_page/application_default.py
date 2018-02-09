# Handling the requests of the webpage
import logging
import json
import uuid
import unirest

from time             import sleep
from threading        import Thread
from logging.handlers import MemoryHandler

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

        # TODO: FINISH THIS PROCESSING
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
logger_handler = logging.FileHandler('/home/brumen/tmp/log1.txt')
logger_handler.setFormatter(logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s'))
logger.addHandler(logger_handler)
computeStream = ComputeStream()  # object keeping the messages
stream_handler = logging.handlers.MemoryHandler( 0  # capcity = 0, immediately flush
                                               , flushLevel = logging.INFO
                                               , target     = computeStream )
logger.addHandler(stream_handler)


from ao_scripts.find_relevant_carriers import get_carrier_l
from ao_scripts.verify_airline         import is_valid_airline
from ao_scripts.verify_origin          import is_valid_origin
from ao_scripts.recompute_option       import recompute_option
from ao_scripts.compute_option         import compute_option

from ao_scripts.ao_auto_fill_origin    import show_airline_l, show_airport_l

from flask import Flask, request, jsonify, Response


app = Flask(__name__)
app.debug = True
app.use_debugger = False
# app.use_reloader = False


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
def recompute_option():

    return recompute_option(request.args)  # TODO: THIS HAS TO CHANGE


@app.route('/write_inquiry', methods=['POST'])
def write_inquiry():

    # TODO LOTS OF TODO HERE
    return 1



@app.route('/compute_option', methods = ['GET'])
def compute_option_flask():
    """
    Computes the option w/ server sent events (SSE)

    """

    # create a new stream for logging
    computeStream = ComputeStream()  # object keeping the messages
    stream_handler = logging.handlers.MemoryHandler( 0  # capcity = 0, immediately flush
                                                   , flushLevel = logging.INFO
                                                   , target     = computeStream)
    logger.addHandler(stream_handler)

    computeThread = Thread(target=compute_option, args=[request.args])  # this is writing to logger
    computeThread.start()  # this thread is computing here

    return Response( computeStream.getMessages()  # this will be returning the messages
                   , mimetype="text/event-stream")


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
