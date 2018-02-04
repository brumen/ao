# Handling the requests of the webpage

from ao_scripts.find_relevant_carriers import get_carrier_l
from ao_scripts.verify_airline         import is_valid_airline
from ao_scripts.verify_origin          import is_valid_origin
from ao_scripts.recompute_option       import recompute_option

from flask import Flask, request, jsonify


app = Flask(__name__)


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

    # TODO LOTS OF TODO HERE
    return recompute_option()  # TODO: THIS HAS TO CHANGE


@app.route('/write_inquiry', methods=['POST'])
def write_inquiry():

    # TODO LOTS OF TODO HERE
    return 1


# TODO: THIS IS SPECIAL - THIS IS SERVER SIDE EVENTS
@app.route('/compute_option')
def compute_option():
    """
    Computes the option w/ server side events

    """

    return 1