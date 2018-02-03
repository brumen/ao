# Auto fill airport destination

import json

from ao_codes import iata_cities_codes, iata_codes_cities


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


def compute_return_l( term
                    , iata_cities_codes =iata_cities_codes
                    , iata_codes_cities =iata_codes_cities ):
    """
    returns the list of places from term 

    """

    search_l = iata_cities_codes.keys()
    search_l.extend(iata_codes_cities.keys())
    ret_cand_1 = filter(lambda x: term.upper() in x, search_l)

    if not ret_cand_1:  # nothing found, report problem
        return []             , False
    elif len(ret_cand_1) < 10:
        return ret_cand_1     , True
    else:
        return ret_cand_1[:10], True

# do the work 
form = cgi.FieldStorage()
term_str = form.getvalue("term")  # this is the string you get
return_l, found_ind = compute_return_l(term_str)
print_for_js(return_l)
