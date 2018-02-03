#!/usr/bin/env python

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
                     iata_cities_codes=iata_cities_codes,
                     iata_codes_cities=iata_codes_cities):
    """
    returns the list of places from term 
    """
    term_upper = term.upper()
    # filter_fct = lambda x: x.startswith(term_upper)
    filter_fct = lambda x: term_upper in x
    search_l = iata_cities_codes.keys()
    search_l.extend(iata_codes_cities.keys())
    #ret_cand_1 = filter(filter_fct, iata_cities_codes.keys())
    ret_cand_1 = filter(filter_fct, search_l)
    if len(ret_cand_1) == 0:  # nothing found, report problem
        return [], False
    elif len(ret_cand_1) < 10:
        return ret_cand_1, True
    else:
        return ret_cand_1[:10], True

# do the work 
form = cgi.FieldStorage()
term_str = form.getvalue("term")  # this is the string you get
return_l, found_ind = compute_return_l(term_str)
print_for_js(return_l)
