#!/usr/bin/env python

# finds the relevant carriers 
import sys
import cgi
import cgitb  # for troubleshooting
import json
import getpass  # for username 
import pandas as pd

# my local modules 
sys.path.append('/home/brumen/work/ao/')
sys.path.append('/home/brumen/public_html/cgi-bin/')
import config
import get_data
import ao_codes
from ao_codes import iata_codes_airlines
import ao_estimate as aoe
cgitb.enable(display=0, logdir=ao_codes.debug_dir)  # for troubleshooting
log_file = ao_codes.error_log + '_' + getpass.getuser()


def print_for_js(is_valid, return_l):
    """
    writes the list of carriers in json format
    """
    body = json.dumps({'is_valid': is_valid,
                       'list_carriers': return_l})

    # this needs to be here so that JSON parser in JavaScript succeeds 
    print "Content-Type: application/json"
    print "Length:", len(body)
    print ""
    print body


def get_carrier_l(origin, dest):
    """
    gets the list from the params database
    """
    # get the three letter codes from origin, dest
    origin_upper, origin_valid = get_data.validate_airport(origin)
    dest_upper, dest_valid = get_data.validate_airport(dest)

    if origin_valid and dest_valid:  # return carriers
        mconn = aoe.constr_mysql_conn()
        exec_q = """
        SELECT DISTINCT(carrier) FROM params 
        WHERE orig = '{0}' AND dest = '{1}'
        """.format(origin_upper, dest_upper)
        df1 = pd.read_sql_query(exec_q, mconn)
        ret_cand_1 = list(df1['carrier'])
        if len(ret_cand_1) == 0:  # no result
            return False, []
        else:  # we have to return all the candidates 
            # extend with flight names
            ret_cand_1.extend([iata_codes_airlines[x] for x in ret_cand_1])
            return True, ret_cand_1
    else:  # wrong inputs
        return False, []
    

# do the work 
form = cgi.FieldStorage()
origin = form.getvalue("origin")
dest = form.getvalue("dest")
is_valid, return_l = get_carrier_l(origin, dest)
print_for_js(is_valid, return_l)
