# updates the carrier field from the origin -> destination (that's the only criterion)
import json

# ao local modules
from ao_codes import iata_airlines_codes, iata_codes_airlines


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


def find_all_carriers( orig
                     , dest
                     , outgoing_date ):
    """
    returns the list of all carriers between orig -> dest on an outgoing_date

    """

    verify(orig)
    verify(dest)
    
    term_upper = term.upper()

    filter_fct = lambda x: x.upper().startswith(term_upper)
    search_l = iata_airlines_codes.keys()
    search_l.extend(iata_codes_airlines.keys())
    ret_cand_1 = filter(filter_fct, search_l)

    return not len(ret_cand_1) == 0


# do the work 
form = cgi.FieldStorage()
orig_str = form.getvalue("orig")
dest_str = form.getvalue("dest")
outgoing_date = form.getvalue("outgoing_date")  # take the first date 
carrier_list = find_all_carriers(orig_str, dest_str, outgoing_date)
print_for_js({'carrier_list': carrier_list})
