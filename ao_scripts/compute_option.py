# Compute script

from contextlib import contextmanager
import time
import sys, os
import os.path
import cgi
import numpy as np
import json

# local path import first 
import config
import ao_codes  # definition files 
log_file = ao_codes.error_log

# ao modules
import air_option as ao
from get_data import get_data
from daemon_local import AoDaemon


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

            
def print_only_progress(is_complete, progress_notice, new_timestamp):
    """
    server response

    """
    ret_dict = { 'is_complete'    : is_complete
               , 'new_timestamp'  : new_timestamp
               , 'progress_notice': progress_notice}

    body = json.dumps(ret_dict)

    # writing the response back 
    print "Content-Type: application/json"
    print "Length:", len(body)
    print ""
    print body


def print_query( valid_inp
               , return_ind
               , output
               , price_range
               , flights_v
               , reorg_flights_v
               , minmax_v ):
    """
    reorganizes the data so that JavaScript can reorganize them better

    """

    ret_dict = {'is_complete'      : True
                # I BELIEVE I DONT NEED THE NEXT 2 LINES
               , 'new_timestamp'   : '2017--'  # irrelevant
               , 'progress_notice' : 'finished'  # also irrelevant
               , 'valid_inp'       : valid_inp
               , 'return_ind'      : return_ind
               , 'price'           : output
               , 'flights'         : flights_v
               , 'reorg_flights'   : reorg_flights_v
               , 'minmax'          : minmax_v
               , 'price_range'     : price_range}

    body = json.dumps(ret_dict)

    # writes to a file
    lt          = time.localtime()
    as_of       = str(lt.tm_year) + '_' + str(lt.tm_mon) + '_' + str(lt.tm_mday) + '_' + \
                  str(lt.tm_hour) + '_' + str(lt.tm_min) + '_' + str(lt.tm_sec)
    inquiry_dir = ao_codes.inquiry_dir + 'inquiry_solo/'
    query_file  = inquiry_dir + 'query_' + as_of + '.txt'

    fo = open(query_file, 'w')
    fo.write(body)
    fo.close()


def print_to_file( valid_inp
                 , return_ind
                 , output
                 , price_range
                 , flights_v
                 , reorg_flights_v
                 , minmax_v
                 , file_used ):
    """
    reorganizes the data so that JavaScript can reorganize them better 

    """

    ret_dict = {'is_complete': True,
                # I BELIEVE I DONT NEED THE NEXT 2 LINES 
                'new_timestamp': '2017--',  # irrelevant
                'progress_notice': 'finished',  # also irrelevant 
                'valid_inp': valid_inp,
                'return_ind': return_ind,
                'price': output,
                'flights': flights_v,
                'reorg_flights': reorg_flights_v,
                'minmax': minmax_v,
                'price_range': price_range}

    body = json.dumps(ret_dict)
    # writes the results in a communicating file 
    fo = open(file_used, 'w')
    fo.write(body)
    fo.close()

    
def compute_price( is_one_way
                 , all_valid
                 , origin_place
                 , dest_place
                 , option_start
                 , option_end
                 , outbound_start
                 , outbound_end
                 , strike
                 , carrier_used
                 , return_ow
                 , cabin_class
                 , nb_people
                 , inbound_start
                 , inbound_end
                 , option_start_ret
                 , option_end_ret
                 , file_used ):
    """
    computes the price and responds to client (both)
    this is the routine forked off in a parent 

    """

    if not all_valid:  # dont compute, inputs are wrong
        print_to_file(False, False, 0, {}, {}, {}, {}, file_used)
        print_query(False, False, 0, {}, {}, {}, {})
    else:
        with suppress_stdout():
            if return_ow == 'one-way':
                result, price_range, flights_v, reorg_flights_v, minmax_v = \
                    ao.compute_option_val(origin_place=origin_place,
                                          dest_place=dest_place,
                                          option_start_date=option_start,
                                          option_end_date=option_end,
                                          outbound_date_start=outbound_start,
                                          outbound_date_end=outbound_end,
                                          carrier=carrier_used,
                                          cabinclass=cabin_class,
                                          adults=nb_people,
                                          K=np.double(strike),
                                          write_data_progress=file_used,
                                          errors='ignore')  # ignore errors here 
            else:
                result, price_range, flights_v, reorg_flights_v, minmax_v = \
                    ao.compute_option_val(origin_place=origin_place,
                                          dest_place=dest_place,
                                          option_start_date=option_start,
                                          option_end_date=option_end,
                                          outbound_date_start=outbound_start,
                                          outbound_date_end=outbound_end,
                                          option_ret_start_date=option_start_ret,
                                          option_ret_end_date=option_end_ret,
                                          inbound_date_start=inbound_start,
                                          inbound_date_end=inbound_end,
                                          carrier=carrier_used,
                                          cabinclass=cabin_class,
                                          adults=nb_people,
                                          K=np.double(strike),
                                          return_flight=True,
                                          write_data_progress=file_used,
                                          errors='ignore')  # ignore errors here

        if result == 'Invalid':
            print_to_file(False, [], -1., [], [], [], [], file_used)  # invalid entries 
            print_query(False, [], -1., [], [], [], [])  # logging only 
        else:  # actual display
            result_ref = str(np.int(result['avg']))
            # logging the query
            is_one_way = return_ow == 'one-way'
            print_to_file(True, return_ow, result_ref, price_range,
                          flights_v, reorg_flights_v, minmax_v, file_used)
            # next is for loggin only 
            print_query(True, return_ow, result_ref, price_range,
                        flights_v, reorg_flights_v, minmax_v)
            

def read_and_reply(file_used, timestamp):
    fo = open(file_used, 'r')
    new_timestamp = os.path.getmtime(file_used)
    progress_notice = fo.read().replace('\n', '')  # file in a string
    fo.close()

    if progress_notice != '':  # file is not empty
        fo = open(file_used, 'r')
        progress_obj = json.load(fo)
        fo.close()
    # progress_notice can be empty string, or a json dump w/ is_complete descriptor

    if timestamp == 'null':  # this special case 
        progress_notice = json.dumps({'is_complete': False,
                                      'progress_notice': 'Initiating flight fetch.'})
        print_only_progress(False, progress_notice, new_timestamp)  # sure to continue
    else:
        if progress_notice != '':  # not empty
            is_complete = progress_obj['is_complete']
        print_only_progress(is_complete, json.dumps(progress_obj), new_timestamp)


# server response logic 
form = cgi.FieldStorage()
data_very_raw = get_data(form)
is_one_way = len(data_very_raw) == 12

if is_one_way:  # one-way flight
    all_valid, origin_place, dest_place, option_start, option_end, \
        outbound_start, outbound_end, strike, carrier_used, return_ow, \
        cabin_class, nb_people = get_data(form)
    inbound_start, inbound_end, option_start_ret, option_end_ret = None, None, None, None
else:
    all_valid, origin_place, dest_place, option_start, option_end, \
        outbound_start, outbound_end, strike, carrier_used, \
        option_start_ret, option_end_ret, inbound_start, inbound_end, \
        return_ow, cabin_class, nb_people = get_data(form)

timestamp = form.getvalue('timestamp')
pcs_id = str(form.getvalue('pcs_id'))  # process id, used for file communication 
file_used = config.prod_dir + 'inquiry/compute/' + pcs_id
if not os.path.exists(file_used):  # create file, otherwise nothing 
    interactive_file = open(file_used, 'a')  # interactive file is emtpy here
    interactive_file.close()

    
if timestamp == 'null':  # daemonize the computation

    # flush null response immediately
    read_and_reply(file_used, 'null')
    sys.stdout.flush()

    # fork the process 
    params = is_one_way, \
             all_valid, origin_place, dest_place, option_start, option_end, \
             outbound_start, outbound_end, strike, carrier_used, return_ow, \
             cabin_class, nb_people, \
             inbound_start, inbound_end, option_start_ret, option_end_ret, \
             file_used
    cod = AoDaemon(compute_price, params)
    cod.start()

else:  # successive attempt at finishing 

    read_and_reply(file_used, timestamp)
