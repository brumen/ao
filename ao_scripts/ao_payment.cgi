#!/usr/bin/env python
import sys
import cgi
import cgitb  # for troubleshooting
cgitb.enable(display=0, logdir="/home/brumen/public_html/inquiry")  # for troubleshooting
import json
import httplib
import uuid 
import unirest

# my local modules 
sys.path.append('/home/brumen/work/ao/')
sys.path.append('/home/brumen/public_html/cgi-bin/')
import config



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

# The following variables need to be assigned:
#   card_nonce
#   location_id
#   access_token
form = cgi.FieldStorage()
card_nonce = form.getvalue("card_nonce")
location_id = form.getvalue("location_id")
access_token = form.getvalue("access_token")

# work that is done 
response = unirest.post('https://connect.squareup.com/v2/locations/' + location_id + '/transactions',
  headers={
    'Accept': 'application/json',
    'Content-Type': 'application/json',
    'Authorization': 'Bearer ' + access_token,
  },
  params = json.dumps({
    'card_nonce': card_nonce,
    'amount_money': {
      'amount': 100,
      'currency': 'USD'
    },
    'idempotency_key': str(uuid.uuid1())
  })
)

print response.body
