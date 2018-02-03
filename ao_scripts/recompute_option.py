#!/usr/bin/env python

# re-computes the option from the forwards given 
from contextlib import contextmanager
import sys, os
import cgi
import numpy as np
import cgitb
import json
import time 
import subprocess

# my local modules 
sys.path.append('/home/brumen/work/ao/')
sys.path.append('/home/brumen/public_html/cgi-bin/')
import config
import ao_codes
cgitb.enable(display=0, logdir=ao_codes.debug_dir)  # for troubleshooting
import ds
import air_option as ao
from get_data import get_data, get_data_dict


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


def print_for_js(valid_inp, return_ind, output, price_range,
                 flights_v, reorg_flights_v, minmax_v,
                 do_nothing=False):
    """
    reorganizes the data so that JavaScript can reorganize them better 
    """
    ret_dict = {'valid_inp': valid_inp,
                'return_ind': return_ind,
                'price': output,
                'flights': flights_v,
                'reorg_flights': reorg_flights_v,
                'minmax': minmax_v,
                'price_range': price_range,
                'do_nothing': do_nothing}

    body = json.dumps(ret_dict)
    print "Content-Type: application/json"
    print "Length:", len(body)
    print ""
    print body


# complete web generation
form = json.load(sys.stdin)  # form is a dict
# form = json.load(open('/home/brumen/public_html/cgi-bin/t1.txt', 'r'))

# this case below is handled in javascript 
# if not form.has_key('flights_selected'):
#    print_for_js(False, '-1', '-1', '-1', '-1', '-1', True)  # dont do anything
# else:
    
data_very_raw = get_data_dict(form)
is_one_way = len(data_very_raw) == 12  # is the flight return or one-way
if is_one_way:  # one-way flight
    all_valid, origin_place, dest_place, option_start, option_end, \
        outbound_start, outbound_end, strike, carrier_used, return_ow, \
        cabin_class, nb_people = data_very_raw
else:
    all_valid, origin_place, dest_place, option_start, option_end, \
        outbound_start, outbound_end, strike, carrier_used, \
        option_start_ret, option_end_ret, inbound_start, inbound_end, \
        return_ow, cabin_class, nb_people = data_very_raw

# flights to select 
sel_flights = form['flights_selected']  # in string form, convert to dict
sel_flights_dict = json.loads(sel_flights)  # in dict form 
lt = time.localtime()
as_of = str(lt.tm_year) + '_' + str(lt.tm_mon) + '_' + str(lt.tm_mday) + '_' + \
        str(lt.tm_hour) + '_' + str(lt.tm_min) + '_' + str(lt.tm_sec)
inquiry_dir = ao_codes.inquiry_dir + 'inquiry_solo/'
inquiry_file = inquiry_dir + 'inquiry_' + as_of + '_from_recompute.inq'


if is_one_way:
    # these 4 below are not important, so they are hardcoded
    option_ret_start_date = '20161212'
    option_ret_end_date = '20161213'
    inbound_date_start = '2016-12-20'
    inbound_date_end = '2016-12-21'


if not all_valid:
    print_for_js(False, False, '-1', {}, {}, {}, {}, True)
else:
    # with suppress_stdout():
    if is_one_way:
        result, price_range, flights_v, reorg_flights_v, minmax_v = \
                ao.compute_option_val(origin_place=origin_place,
                                      dest_place=dest_place,
                                      flights_include=sel_flights_dict,
                                      option_start_date=option_start,
                                      option_end_date=option_end,
                                      outbound_date_start=outbound_start,
                                      outbound_date_end=outbound_end,
                                      carrier=carrier_used,
                                      cabinclass=cabin_class,
                                      adults=np.int(nb_people),
                                      K=np.double(strike),
                                      recompute_ind=True)
    else:
        result, price_range, flights_v, reorg_flights_v, minmax_v = \
                ao.compute_option_val(origin_place=origin_place,
                                      dest_place=dest_place,
                                      flights_include=sel_flights_dict,
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
                                      adults=np.int(nb_people),
                                      K=np.double(strike),
                                      return_flight=True,
                                      recompute_ind=True)

    if result == 'Invalid':
        print_for_js(False, False, '-1', [], [], [], [], True)  # invalid entries
    else:
        # THIS IS WRONG 
        # result_ref = str(np.int(result['avg'])) - DOESNT WORK, NOT IMPORTANT 
        result_ref = str(np.int(1.))

        print_for_js(True, return_ow, result_ref, price_range,
                     flights_v, reorg_flights_v, minmax_v, False)
        # write an inquiry in the inquiry log 
        res_write = {'origin_place': origin_place,
                     'dest_place': dest_place,
                     'flights_include': sel_flights_dict,
                     'option_start_date': option_start,
                     'option_end_date': option_end,
                     'outbound_date_start': outbound_start,
                     'outbound_date_end': outbound_end,
                     'carrier': carrier_used,
                     'cabinclass': cabin_class,
                     'adults': nb_people,
                     'K': np.double(strike)}
        if not is_one_way:
            res_write['option_ret_start_date'] =  option_start_ret
            res_write['option_ret_end_date'] = option_end_ret
            res_write['inbound_date_start'] = inbound_start
            res_write['inbound_date_end'] = inbound_end
        fo = open(inquiry_file, 'w')
        fo.write(json.dumps(res_write))
        fo.close()
