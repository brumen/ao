""" Rester service to handle the request from the webpage
"""

import datetime
import logging
import json
import os.path
import sys
sys.path.append('/home/brumen/work/')

from time  import localtime
from flask import Flask, request, jsonify, Response, render_template
from json  import dumps

from ao.config                    import LOG_DIR, LOG_FILE
from ao.ao_codes                  import inquiry_dir
from ao.iata.codes                import get_airline_code, get_city_code
from ao.ao_scripts.verify_origin  import get_carriers_on_route
from ao.ao_scripts.compute_option import compute_option


logging.basicConfig( filename = os.path.join(LOG_DIR, LOG_FILE)
                   , level    = logging.CRITICAL)


logger = logging.getLogger(__name__)


# Rester Flask service
ao_rester = Flask(__name__)
ao_rester.debug = True
ao_rester.use_debugger = False


def time_now() -> str:
    """ Returns local time in the string format.

    :returns: local time separated by underscores.
    """

    lt = localtime()

    return '_'.join([ str(lt.tm_year)
                    , str(lt.tm_mon)
                    , str(lt.tm_mday)
                    , str(lt.tm_hour)
                    , str(lt.tm_min)
                    , str(lt.tm_sec) ] )


@ao_rester.route('/ao/')
def hello():
    """ For testing purposes only.
    """

    return 'Hello World'


@ao_rester.route('/ao/test_2')
def hello2():
    """ Demonstration of the SSE.
    """

    def stream1():
        """ structure has to be id: \nevent: \ndata: \nn
        """
        yield """id: 1\nevent: help11\ndata: {"help22": 2}\n\n"""
        yield """id: 2\nevent: help11\ndata: {"help22": 2}\n\n"""

    return Response(stream1(),  mimetype='text/event-stream')


@ao_rester.route('/ao/<name>')
def test_3(name):
    return render_template('user.html', name = name)


@ao_rester.route('/ao/verify_airline', methods = ['GET'])
def verify_airline():
    """ Checks that the airline is correct.

    arguments provided in the request should be:
        airline: e.g. like Adria, it will return JP
    """

    # arguments is of type ImmutableMultiDict, one of the parameters should be airline
    arguments = request.args

    if 'airline' not in arguments:
        return jsonify({'valid': False})

    return jsonify({ 'valid'  : True
                   , 'airline': get_airline_code(request.args.get('airline'))})


@ao_rester.route('/ao/verify_origin', methods = ['GET'])
def verify_origin():
    """ Checks whether the airport is a valid IATA name.
        arguments provided in request are:
            origin: e.g. airport number 'EWR'
    """

    arguments = request.args
    if 'origin' not in arguments:
        return jsonify({'valid' : False})

    return jsonify ({ 'valid': True
                    , 'city_code': get_city_code(arguments.get('origin'))})


@ao_rester.route('/ao/find_relevant_carriers', methods = ['GET'])
def find_relevant_carriers():
    """ Endpoint for finding the relevant carriers from origin to dest.
        request arguments:
           origin: origin IATA airport, e.g. 'EWR'

    """

    arguments = request.args

    if ('origin' not in arguments) or ('dest' not in arguments):
        return jsonify({'valid': False})

    return jsonify({ 'valid'        : True
                   , 'list_carriers': get_carriers_on_route( arguments.get('origin'), arguments.get('dest'))
                   , } )


@ao_rester.route('/ao/recompute_option', methods = ['POST'])
def recompute_option():
    """ Recomputes the option value,
           makes the same call as the compute_option with the additional flag.
           No need for publisher here, as this is fast.
    """

    return jsonify(compute_option( request.get_json(), recompute_ind = True ) )


@ao_rester.route('/ao/write_inquiry', methods=['POST'])
def write_inquiry():
    """ Writes a file about the inquiry to the inquiry folder.

    """

    with open(f'{inquiry_dir}/inquiry_solo/inquiry_{time_now()}.inq', 'w') as fo:
        fo.write(json.dumps(request.get_json()))

    # succeeds, returns true
    return jsonify({'valid': True})


@ao_rester.route('/ao/compute_option', methods = ['GET'])
def compute_option_all():
    """ Computes the option w/ server sent events (SSE)
    """

    # compute_option has to be a generator
    ''' Request is a form of:
        { 'return_ow': string 'one_way' or 'return'
        , 'cabin_class': string
        , 'nb_people': integer
        , 'origin_place': IATA origin code, 3 letter code
        , 'dest_place': IATA destination, 3 letter code
        , 'option_start': datetime.date
        , 'option_end'    ] = document.getElementById("option-end-date").value;
        , 'outbound_start'] = document.getElementById("js-depart-input").value;
        , 'outbound_end'  ] = document.getElementById("js-return-input").value;
        , 'ticket_price'  ] = document.getElementById("ticket-price").value;
        , 'airline_name'  ] = document.getElementById("airline-name").value;

    if (return_ow == 'return')
	// get the return data
	'option_ret_start': document.getElementById("option-start-date-return").value;
	'option_ret_end'     = document.getElementById("option-end-date-return").value;
	'outbound_start_ret' = document.getElementById("js-depart-input-return").value;
	'outbound_end_ret'   = document.getElementById("js-return-input-return").value;
    '''

    # arguments = json.loads(request.data.decode('utf8'))
    arguments = request.args

    # TODO: This is a trick for testing now
    market_date = datetime.date(2017, 4, 26)

    def compute_for_response(arguments, market_date=datetime.date(2017, 4, 26)):
        """ Modifying the responses from the compute_option function.
        """

        counter = 1
        try:
            for msg in compute_option(arguments, market_date=market_date):  # a generator
                yield f"id: {counter}\ndata: {dumps(msg)}\n\n"  # appropriate formatting string.
                counter += 1
        except StopIteration as si:
            pass

    # compute_option has to be a generator for sse events
    return Response(compute_for_response(arguments, market_date=market_date), mimetype='text/event-stream')


@ao_rester.route('/ao/ao_auto_fill_origin')
def ao_auto_fill_origin():
    """ Returns the auto fill of the IATA origin airports
    """

    airports = get_city_code(request.args.get("term"))

    return Response( json.dumps(airports)  # jsonify doesnt work
                   , mimetype = 'application/json')


@ao_rester.route('/ao/ao_auto_fill_airline')
def ao_auto_fill_airline():
    """ Returns the auto fill of the IATA airline codes.
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

ao_rester.run()

# testing:
# try in browser: http://localhost:5000/ao/verify_airline?airline=Adria
# for pricing w/ iterators.
# http://localhost:5000/ao/compute_option?origin=San%20Francisco&dest=Newark&outbound_start=05/10/2017&outbound_end=05/15/2017&airline_name=United&return_ow=one_way&cabin_class=Economy&nb_people=1&ticket_price=100