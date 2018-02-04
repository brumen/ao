# Handling the requests of the webpage
from time import sleep

from ao_scripts.find_relevant_carriers import get_carrier_l
from ao_scripts.verify_airline         import is_valid_airline
from ao_scripts.verify_origin          import is_valid_origin
from ao_scripts.recompute_option       import recompute_option
from ao_scripts.compute_option         import compute_option

from ao_scripts.ao_auto_fill_origin    import show_airline_l, show_airport_l

from flask import Flask, request, jsonify, Response


app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/multiple')
def multiple_messages():

    def eventStream():
        while True:
            # Poll data from the database
            # and see if there's a new message
            sleep(5)
            yield "data: {0}\n\n".format("silly")

    return Response(eventStream(), mimetype="text/event-stream")


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

    # TODO LOTS OF TODO HERE
    return recompute_option()  # TODO: THIS HAS TO CHANGE


@app.route('/write_inquiry', methods=['POST'])
def write_inquiry():

    # TODO LOTS OF TODO HERE
    return 1



@app.route('/compute_option', methods = ['GET'])
def compute_option():
    """
    Computes the option w/ server sent events (SSE)

    """

    def computeOptionEventStream():
        comp1 = compute_option(request.form)
        while True:
            # Poll data from the database
            # and see if there's a new message
            sleep(5)
            yield "data: {0}\n\n".format("silly")

    return Response(computeOptionEventStream(), mimetype="text/event-stream")


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
