#!/usr/bin/env python

import sys
import cgi
import cgitb  # for troubleshooting
import json

# my local modules 
sys.path.append('/home/brumen/work/ao/')
sys.path.append('/home/brumen/public_html/cgi-bin/')
import config
import ao_codes
from ao_codes import iata_cities_codes, iata_codes_cities, iata_airlines_codes, iata_codes_airlines
cgitb.enable(display=0, logdir=config.prod_dir + "debug/")  # for troubleshooting


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


def is_valid_name(term,
                  iata_cities_codes=iata_cities_codes,
                  iata_codes_cities=iata_codes_cities):
    """
    returns the list of places from term 
    """
    if term is None:
        return False
    term_upper = term.upper()
    filter_fct = lambda x: x.startswith(term_upper)
    search_l = iata_cities_codes.keys()
    search_l.extend(iata_codes_cities.keys())
    #ret_cand_1 = filter(filter_fct, iata_cities_codes.keys())
    ret_cand_1 = filter(filter_fct, search_l)
    return not len(ret_cand_1) == 0

# do the work 
form = cgi.FieldStorage()
term_str = form.getvalue("origin")  # check if this name is valid 
found_ind = is_valid_name(term_str)
print_for_js({'found': found_ind})
