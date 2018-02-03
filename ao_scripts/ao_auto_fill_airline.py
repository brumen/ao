# Find airline

import json

from ao_codes import iata_airlines_codes


def print_for_js(return_l):
    """
    writes the list in json format

    :param return_l: TODO:
    :type return_l:
    """
    body = json.dumps(return_l)
    # this needs to be here so that JSON parser in JavaScript succeeds 
    print "Content-Type: application/json"
    print "Length:", len(body)
    print ""
    print body


def compute_return_l( term
                    , iata_airlines_codes = iata_airlines_codes ):
    """
    returns the list of places from term

    """

    ret_cand_1 = filter( lambda x: term.upper() in x.upper()
                       , iata_airlines_codes.keys())

    if len(ret_cand_1) < 10:
        return ret_cand_1
    else:
        return ret_cand_1[:10]

# do the work 
form = cgi.FieldStorage()

term_str = form.getvalue("term")  # this is the string you get

return_l = compute_return_l(term_str)

print_for_js(return_l)
