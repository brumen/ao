#!/usr/bin/env python
from contextlib import contextmanager
import time
import sys, os
import os.path
import cgi
import numpy as np
import json
import cgitb  # cgi troubleshoot

# local path import first 
sys.path.append('/home/brumen/work/zo/')
sys.path.append('/home/brumen/public_html/cgi-bin/')
import config
import zo_codes  # definition files 
cgitb.enable(display=0, logdir=zo_codes.debug_dir)  # for troubleshooting
log_file = zo_codes.error_log
# ao modules 
import zo
import ds
from get_zo_data import get_zo_data

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


def print_query(valid_inp, option_value, details):
    """
    reorganizes the data so that JavaScript can reorganize them better 
    """
    ret_dict = {'is_complete': True,
                'valid_inp': valid_inp,
                'price': option_value,
                'details': details}
    body = json.dumps(ret_dict)

    # writes to a file
    # lt = time.localtime()
    # as_of = str(lt.tm_year) + '_' + str(lt.tm_mon) + '_' + str(lt.tm_mday) + '_' + \
    #         str(lt.tm_hour) + '_' + str(lt.tm_min) + '_' + str(lt.tm_sec)
    # inquiry_dir = ao_codes.inquiry_dir + 'inquiry_solo/'
    # query_file = inquiry_dir + 'query_' + as_of + '.txt'
    # fo = open(query_file, 'w')
    # fo.write(body)
    # fo.close()

    # this needs to be here so that JSON parser in JavaScript succeeds 
    print "Content-Type: application/json"
    print "Length:", len(body)
    print ""
    print body

    
def compute_price(address, city, zipcode, state, expiry, strike):
    """
    computes the zoption price and responds to client (both)
    """
    address_used = address + ', ' + city + ', ' + state
    lt = time.localtime()
    date_today = str(lt.tm_year) + str(ds.d2s(lt.tm_mon)) + str(ds.d2s(lt.tm_mday))
    date_today_dt = ds.convert_str_date(date_today)
    expiry_dt = ds.convert_str_date(ds.convert_dateslash_str(expiry))
    expiry_used = (expiry_dt - date_today_dt).days / 365.25  # TO IMPROVE TO IMPROVE 
    with suppress_stdout():
        is_valid, data, option_value = zo.price_option_addr(address_used, zipcode,
                                                            expiry=expiry_used,
                                                            K=np.double(strike))
    eo_log = open(log_file, 'a')
    # eo_log = open(str(data), 'a')
    eo_log.write('eee ' + str(is_valid))
    eo_log.write('\n')
    eo_log.close()

    if is_valid == 'valid': 
        home_details = data['links']['home_details']
        return is_valid, home_details, option_value
    else:
        return is_valid, None, -1.


# server response logic 
form = cgi.FieldStorage()
address, city, zipcode, state, expiry, strike = get_zo_data(form)

eo_log = open(log_file, 'a')
eo_log.write('ccc ')
eo_log.write('\n')
eo_log.close()


is_valid, home_details, option_value = compute_price(address, city, zipcode, state, expiry, strike)

eo_log = open(log_file, 'a')
eo_log.write('fff ' + str(is_valid))
eo_log.write('\n')
eo_log.close()


if is_valid == 'valid':
    print_query(True, str(np.int(option_value)), home_details)
else:  # invalid 
    print_query(False, str(-1), home_details)
