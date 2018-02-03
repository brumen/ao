from contextlib import contextmanager
import sys, os
import cgi
import cgitb
import json
from dateutil.parser import parse

# ao modules 
sys.path.append('/home/brumen/work/zo/')
sys.path.append('/home/brumen/public_html/cgi-bin/')
import config
import zo
import ds
cgitb.enable(display=0, logdir=config.prod_dir + "debug/")  # for troubleshooting


def validate_outbound_dates(date_):
    """
    validates whether date is in the 1/2/2017 format, and converts it into - format 
    """
    try:
        parse(date_)
    except ValueError:
        return "", False
    return ds.convert_dateslash_dash(date_), True  # if everything OK, then convert


def validate_strike(strike_i):
    try:
        float(strike_i)
        return float(strike_i), True
    except ValueError:
        return -1., False

    
def get_zo_data(form):
    """
    obtains data from the form and returns them 
    """
    address = form.getvalue('address')
    city = form.getvalue('city')
    zipcode = form.getvalue('zipcode')
    state = form.getvalue('state')
    expiry = form.getvalue('expiry')
    strike = form.getvalue('strike')

    return (address, city, zipcode, state, expiry, strike)
