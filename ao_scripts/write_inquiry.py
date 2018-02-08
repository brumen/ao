# AirOption inquiry module
import sys

import json
import time 
import subprocess


def display_flights(all_flights, fo):
    """
    display flights

    :param fo: file where to write the flights 
    :param all_flights: flights info
    """
    ow_ind = all_flights['is_one_way']
    outbound_flights = all_flights['outbound']
    if not ow_ind:
        inbound_flights = all_flights['inbound']

    fo.write('Outbound flights: ' + outbound_flights + '\n')
    if not ow_ind:
        fo.write('Inbound flights: ' + inbound_flights + '\n')


def write_inquiry_raw(is_one_way, origin_place, dest_place,
                      outbound_start, outbound_end,
                      option_start, option_end,
                      inbound_start=None, inbound_end=None,
                      option_start_ret=None, option_end_ret=None,
                      strike='-1', carrier='None', 
                      price=None, message_str=None,
                      all_flights=None):
    lt = time.localtime()
    as_of = str(lt.tm_year) + '_' + str(lt.tm_mon) + '_' + str(lt.tm_mday) + '_' + \
            str(lt.tm_hour) + '_' + str(lt.tm_min) + '_' + str(lt.tm_sec)
    inquiry_dir = '/home/brumen/public_html/inquiry/'
    inquiry_file = inquiry_dir + 'inquiry_' + as_of + '.inq'
    inquiry_pdf = inquiry_dir + 'inquiry_' + as_of + '.pdf'
    fo = open(inquiry_file, 'w')
    if is_one_way:
        fo.write('One-way flight\n')
    else:
        fo.write('Return flight\n')
    fo.write('As of: ' + as_of + '\n')
    fo.write('From: ' + origin_place + '\n')
    fo.write('To: ' + dest_place + '\n')
    fo.write('Departure start: ' + outbound_start + '\n')
    fo.write('Departure end: ' + outbound_end + '\n')
    fo.write('Option start: ' + option_start + '\n')
    fo.write('Option end: ' + option_end + '\n')
    if not is_one_way:
        fo.write('Return start: ' + inbound_start + '\n')
        fo.write('Return end: ' + inbound_end + '\n')
        fo.write('Option return start: ' + option_start_ret + '\n')
        fo.write('Option return end: ' + option_end_ret + '\n')

    fo.write('Strike: ' + strike + '\n')
    fo.write('Carrier: ' + carrier + '\n')
    if price is not None:
        fo.write('Price: ' + price + "\n")
    if message_str is not None:
        fo.write('Message: ' + message_str + '\n')
    # display forward values and flights 
    # if all_flights != None:
    #    fo.write('Flights considered\n')
    #    display_flights(all_flights, fo)
    fo.close()
    # compile the text format to pdf
    subprocess.call(['enscript', inquiry_file, '-o', inquiry_pdf])

    
# complete web generation
# form = cgi.FieldStorage()
form = json.load(sys.stdin)  # post form is a dict 
lt = time.localtime()
as_of = str(lt.tm_year) + '_' + str(lt.tm_mon) + '_' + str(lt.tm_mday) + '_' + \
        str(lt.tm_hour) + '_' + str(lt.tm_min) + '_' + str(lt.tm_sec)
inquiry_dir = '/home/brumen/public_html/inquiry/inquiry_solo/'
inquiry_file = inquiry_dir + 'inquiry_' + as_of + '.inq'
# inquiry_pdf = inquiry_dir + 'inquiry_' + as_of + '.pdf'
fo = open(inquiry_file, 'w')
fo.write(json.dumps(form))
fo.close()

# print reutrn 
body = json.dumps({'valid': True})
print "Content-Type: application/json"
print "Length:", len(body)
print ""
print body
