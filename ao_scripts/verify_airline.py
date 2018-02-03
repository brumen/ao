# Verify origin airport
import json

# AirOption modules
from ao_codes import iata_airlines_codes as IATA_AIRLINES_CODES,\
                     iata_codes_airlines as IATA_CODES_AIRLINES


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


def is_valid_name(term):
    """
    returns the list of places from term

    """

    if term is None:
        return False

    search_l = IATA_AIRLINES_CODES.keys()
    search_l.extend(IATA_CODES_AIRLINES.keys())
    ret_cand_1 = filter( lambda x: x.upper().startswith(term.upper())
                       , search_l)
    return not len(ret_cand_1) == 0  # TODO: THIS CAN BE RELAXED

# do the work 
form = cgi.FieldStorage()
term_str = form.getvalue("airline")  # check if this name is valid 
found_ind = is_valid_name(term_str)
print_for_js({'found': found_ind})
