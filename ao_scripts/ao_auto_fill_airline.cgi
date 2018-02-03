#!/usr/bin/env python

# from contextlib import contextmanager
import sys
import cgi
import cgitb  # for troubleshooting
cgitb.enable(display=0, logdir="/home/brumen/public_html/inquiry")  # for troubleshooting
import json

# my local modules 
sys.path.append('/home/brumen/work/ao/')
sys.path.append('/home/brumen/public_html/cgi-bin/')
import config
from ao_codes import iata_cities_codes, iata_codes_cities, iata_airlines_codes, iata_codes_airlines


def print_for_js(return_l):
    """
    writes the list in json format
    """
    body = json.dumps(return_l)
    # this needs to be here so that JSON parser in JavaScript succeeds 
    print "Content-Type: application/json"
    print "Length:", len(body)
    print ""
    print body


def compute_return_l(term,
                     iata_airlines_codes=iata_airlines_codes):
    """
    returns the list of places from term 
    """
    term_upper = term.upper()
    # filter_fct = lambda x: x.upper().startswith(term_upper)
    filter_fct = lambda x: term_upper in x.upper()
    ret_cand_1 = filter(filter_fct, iata_airlines_codes.keys())
    if len(ret_cand_1) < 10:
        return ret_cand_1
    else:
        return ret_cand_1[:10]

# do the work 
form = cgi.FieldStorage()
term_str = form.getvalue("term")  # this is the string you get
return_l = compute_return_l(term_str)
print_for_js(return_l)
