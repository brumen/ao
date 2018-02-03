import json

# my local modules 
from ao_codes import iata_cities_codes as IATA_CITIES_CODES,\
                     iata_codes_cities as IATA_CODES_CITIES


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
    term_upper = term.upper()
    filter_fct = lambda x: x.startswith(term_upper)
    search_l = IATA_CITIES_CODES.keys()
    search_l.extend(IATA_CODES_CITIES.keys())
    #ret_cand_1 = filter(filter_fct, iata_cities_codes.keys())
    ret_cand_1 = filter(filter_fct, search_l)

    return not len(ret_cand_1) == 0

# do the work 
form = cgi.FieldStorage()
term_str = form.getvalue("origin")  # check if this name is valid 
found_ind = is_valid_name(term_str)
print_for_js({'found': found_ind})
