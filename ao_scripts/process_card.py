# process card module
import config
from __future__ import print_function
import os
import uuid
import json
import time
import numpy as np
import getpass

#Jinja2
import jinja2

# for sending e-mail w/ pdf attached
import smtplib
import email
import email.mime.application

# square modules
from squareconnect.rest                 import ApiException
from squareconnect.apis.transaction_api import TransactionApi

# my local modules
import ds
import ao_codes
import air_option as ao

from ao_codes import iata_codes_cities, iata_cities_codes
from ao_codes import airoptions_gmail_pass, airoptions_gmail_acct
from get_data import get_data_final


def check_inputs( card_name
                , sq_postal_code
                , city_state ):
    """
    Checks that the card_name, sq_postal_code and city are alphanumeric

    :param card_name:

    :param sq_postal_code:
    :type sq_postal_code:
    :param city_state:
    :type city_state:
    """

    city_state_alnum = city_state.replace(",", "")  # city, state can be New York, NY

    inp_alnum = card_name.isalnum() and sq_postal_code.isalnum() and \
                city_state_alnum.isalnum()

    return inp_alnum


def process_orig_dest(orig_raw, dest_raw):
    """
    processes the origin/destination in case you have 3 letters for orig_raw/dest_raw
    """
    if len(orig_raw) == 3:
        orig_long = iata_codes_cities[orig_raw]
        orig_short = orig_raw
    else:
        orig_long = orig_raw
        orig_short = iata_cities_codes[orig_raw]
        
    if len(dest_raw) == 3:
        dest_long = iata_codes_cities[dest_raw]
        dest_short = dest_raw
    else:
        dest_long = orig_raw
        dest_short = iata_cities_codes[dest_raw]

    return orig_long, dest_long, orig_short, dest_short


def process_flights(orig, dest, flights_d, fo):
    """
    processes the flights, write everything in a nice pdf file 
    :param orig: originating airport 
    :param dest: destination airport 
    :param flights_d: dict of reorg_flights 
    :param fo: file where the flights are written
    """
    # construct number of flights inventory by days 
    flights_d_nb = dict()
    for day in flights_d:
        flights_d_nb[day] = 0
        if day != 'minmax':  # day: 2017-01-01
            for tod in flights_d[day]:
                if tod != 'minmax':
                    for flight in flights_d[day][tod]:  
                        used_arr = flights_d[day][tod][flight]  # flight is a list 
                        if (flight != "min_max") and used_arr[7]:  # used_arr[7] ... checked if selected
                            flights_d_nb[day] += 1

    inserted_date = """
    <li class="date-item">
    {0}: {1} - {2} ({3} flights)
    </li>
    """
    # flights in dict format, presents them 
    for day in flights_d:
        nb_flights = flights_d_nb[day]
        fo.write(inserted_date.format(day, orig, dest, nb_flights))  # writes the date 
        if day != 'minmax':  # day: 2017-01-01
            for tod in flights_d[day]:
                if tod != 'minmax':
                    for flight in flights_d[day][tod]:  
                        used_arr = flights_d[day][tod][flight]  # flight is a list 
                        if (flight != "min_max") and used_arr[7]:  # used_arr[7] ... checked if selected 
                            # airline & flight_nb (UA13)
                            _, dep_date, dep_time, arr_date, arr_time, _, airline, _ = used_arr 
                            dep_hour, dep_min, dep_sec = dep_time.split(':')
                            dep_time_display = dep_hour + ':' + dep_min
                            # dep_date = ds.convert_datedash_date(dep_time)
                            dep_dt = ds.convert_datedash_time_dt(dep_date, dep_time)
                            arr_hour, arr_min, arr_sec = arr_time.split(':')
                            arr_time_display = arr_hour + ':' + arr_min
                            arr_dt = ds.convert_datedash_time_dt(arr_date, arr_time)
                            duration_sec = (arr_dt - dep_dt).seconds
                            duration_hour = duration_sec / 3600
                            duration_min = (duration_sec - duration_hour * 3600) / 60
                            duration = str(duration_hour) + 'h' + str(duration_min) + 'min'
                            fo.write(insert_flight(airline, orig, dest, dep_time_display,
                                                   arr_time_display, duration))


def insert_flight( airline
                 , orig
                 , dest
                 , orig_time
                 , dest_time
                 , time_diff ):
    """
    generates the entry field for the difference

    """
    # text to be inserted for each flight 
    # {0} ... airline name
    # {1} ... origin
    # {2} ... dest
    # {3} ... dep. time
    # {4} ... arr. time
    # {5} ... duration 

    inserted_text = """
    <li class="day-list-item clearfix ">
    <article data-cid="model_18429" data-deeplink="details" class="card result clearfix no-details  " ontouchstart="">
    <div class="card-body clearfix">
     <div class="clearfix carrier">
       <div class="airline">
         <span>{0}</span>
       </div>
     </div>
     <section data-id="0" class="card-main leg clearfix dept">
       <div class="leg-details long-date-format ">
          <div class="depart">
            <span class="station-tooltip" data-id="ORIGIN_NB">
              <span class="times">{3}</span>
              <span class="stop-station" data-id="ORIGIN_NB">{1}</span>
            </span>
          </div>
          <div class="stops">
          <span class="duration">{5}</span>
          <ul class="stop-line">
            <li class="stop-line"></li>
          </ul>
          <div class="leg-stops no-stops">
            <span class="leg-stops-green leg-stops-label">Non-stop</span>
            <span class="leg-stops-station">
            </span>
          </div>
        </div>
        <div class="arrive">
          <span class="station-tooltip" data-id="DEST_NB">
          <span class="times">{4}</span>
          <span class="stop-station" data-id="DEST_NB">{2}</span>
          </span>
        </div>
      </div>
    </section>
    </div>
    </article>
    </li>
    """

    return inserted_text.format(airline, orig, dest, orig_time, dest_time, time_diff)


def write_invoice(orig, dest, flights_d, fo):
    """
    writes the invoice into the .tex file 
    :param orig: originating airport 
    :param dest: destination airport 
    :param flights_d: dict of reorg_flights 
    :param fo: file where the flights are written
    """
    # 0 - date
    # 1 - flight nb
    # 2 - from
    # 3 - to
    # 4 - departure time
    # 5 - arrival time
    inserted_tabline = """
    \hline
    {0}  & {1}  & {2}  & {3}  & {4} & {5} \\\\
    """
    # flights in dict format, presents them 
    for day in flights_d:
        flight_ct = 0
        if day != 'minmax':  # day: 2017-01-01
            for tod in flights_d[day]:
                if tod != 'minmax':
                    for flight in flights_d[day][tod]:  
                        used_arr = flights_d[day][tod][flight]  # flight is a list 
                        if (flight != "min_max") and used_arr[7]:  # used_arr[7] ... checked if selected 
                            _, dep_date, dep_time, arr_date, arr_time, _, airline, _ = used_arr 

                            if flight_ct == 0:
                                dep_date_used = dep_date
                            else:
                                dep_date_used = ""
                            fo.write(inserted_tabline.format(dep_date_used, airline,
                                                             orig, dest,
                                                             dep_time, arr_time))
                            flight_ct += 1


def write_invoice_fct( invoice_template
                     , invoice_fo
                     , orig_long
                     , dest_long
                     , orig_short
                     , dest_short
                     , card_name
                     , card_address
                     , city_state
                     , nb_people
                     , cabin_class
                     , sq_postal_code
                     , strike
                     , amount_charged
                     , return_ow_final
                     , option_end
                     , option_end_ret
                     , flights_d
                     , invoice_vars
                     , inquiry_logs_dir ):
    """
    write the invoice function

    :param invoice_template: open file containing the template 
    :param invoice_fo: invoice file where to write 
    """
    for line in invoice_template:
        if "ORIGIN" in line:
            invoice_fo.write(str(orig_long))
        elif "DESTINATION" in line:
            invoice_fo.write(str(dest_long))
        elif "DATE" in line:
            newline = line.replace('DATE', as_of.replace('_', '/'))
            invoice_fo.write(newline)
        elif "CUSTOMERNAME" in line:
            newline = line.replace('CUSTOMERNAME', card_name)
            invoice_fo.write(newline)
        elif "CUSTOMERSTREET" in line:
            newline = line.replace('CUSTOMERSTREET', card_address)
            invoice_fo.write(newline)
        elif "CUSTOMERCITY" in line:
            newline = line.replace('CUSTOMERCITY', city_state)
            invoice_fo.write(newline)
        elif "NBADULTS" in line:
            newline = line.replace('NBADULTS', nb_people)
            invoice_fo.write(newline)
        elif "CLASSTRAVEL" in line:
            newline = line.replace('CLASSTRAVEL', cabin_class)
            invoice_fo.write(newline)
        elif "CUSTOMERZIP" in line:
            newline = line.replace('CUSTOMERZIP', sq_postal_code)
            invoice_fo.write(newline)
        elif "TICKETPRICE" in line:
            newline = line.replace('TICKETPRICE', str(strike))
            invoice_fo.write(newline)
        elif 'PRICETOTAL' in line:
            newline = line.replace('PRICETOTAL', str(amount_charged/100))
            invoice_fo.write(newline)
        elif "RETURNONEWAY" in line:
            newline = line.replace('RETURNONEWAY', return_ow_final)
            invoice_fo.write(newline)
        elif "CHANGEOPTIONTEXT" in line:
            if ow_ind:  # one-way
                changed_text = str(ds.convert_str_dateslash(option_end))
            else:  # return 
                changed_text = str(ds.convert_str_dateslash(option_end)) + \
                               ' and until ' + str(ds.convert_str_dateslash(option_end_ret)) + \
                               ' for the return flight'
            newline = line.replace('CHANGEOPTIONTEXT', changed_text)
            invoice_fo.write(newline)
        elif "%INVENTORY" in line:
            if ow_ind:
                write_invoice(orig_short, dest_short, flights_d, invoice_fo)
            else:
                write_invoice(orig_short, dest_short, flights_d[0], invoice_fo)
                write_invoice(dest_short, orig_short, flights_d[1], invoice_fo)
        else:
            invoice_fo.write(line)
    invoice_fo.close()

    # check if the file invoice_tex is alfanumeric
    invoice_tex, invoice_aux, invoice_log, invoice_pdf = invoice_vars
    # write to 
    os.system('pdflatex -output-directory ' + inquiry_dir + ' ' + invoice_tex + ' > /dev/null')
    # move the .aux .tex .log files
    os.system('mv ' + invoice_tex + ' ' + inquiry_logs_dir)
    os.system('mv ' + invoice_aux + ' ' + inquiry_logs_dir)
    os.system('mv ' + invoice_log + ' ' + inquiry_logs_dir)
    os.system('cp ' + invoice_pdf + ' ' + inquiry_logs_dir)  # copy the pdf here as well
    # MISSING THE ELSE CLAUSE 

    
def write_file_fct( fo
                  , template
                  , orig_long
                  , dest_long
                  , orig_short
                  , dest_short
                  , invoice_pdf_loc
                  , amount_charged
                  , nb_people
                  , cabin_class
                  , return_ow_final
                  , flights_d ):
    """

    """

    # write result to the file
    fo.write('Content-type:text/html\r\n\r\n\n')
    for line in template:
        if "ORIGIN_REPLACE" in line:
            fo.write(str(orig_long))
        elif "DEST_REPLACE" in line:
            fo.write(str(dest_long))
        elif 'LOCATION' in line:
            newline = line.replace('LOCATION', invoice_pdf_loc)
            fo.write(newline)
        elif 'PRICE_TOTAL' in line:
            newline = line.replace('PRICE_TOTAL', str(amount_charged/100))
            fo.write(newline)
        elif "NBADULTS" in line:
            newline = line.replace('NBADULTS', str(nb_people))
            fo.write(newline)
        elif "CLASSTRAVEL" in line:
            newline = line.replace('CLASSTRAVEL', str(cabin_class))
            fo.write(newline)
        elif "RETURNONEWAY" in line:
            newline = line.replace('RETURNONEWAY', return_ow_final)
            fo.write(newline)
        elif "TO_REPLACE" in line:
            if ow_ind:
                process_flights(orig_short, dest_short, flights_d, fo)
            else:
                process_flights(orig_short, dest_short, flights_d[0], fo)  # outbound flights
                process_flights(dest_short, orig_short, flights_d[1], fo)  # inbound flights
        else:
            fo.write(line)
    fo.close()


def send_email_to_client( client_email
                        , invoice_pdf ):
    """
    Sends the e-mail to client

    :param client_email: email of the client
    :type client_email: str
    :param invoice_pdf: filename where the generated pdf is stored
    :type invoice_pdf: str
    """

    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.ehlo()
    server.login(airoptions_gmail_acct, airoptions_gmail_pass)

    # plain text part of the message
    msg = email.mime.Multipart.MIMEMultipart()
    msg['Subject'] = 'AirOptions purchase'
    msg['From'   ] = 'airoptions.llc@gmail.com'
    msg['To'     ] = client_email
    body = email.mime.Text.MIMEText("""Thank you for your purchase with AirOptions. 
    Your receipt is attached to this e-mail.""")
    msg.attach(body)

    # pdf attachment
    with open(invoice_pdf, 'rb') as fp:
        att = email.mime.application.MIMEApplication(fp.read(), _subtype="pdf")

    att.add_header( 'Content-Disposition'
                  , 'attachment'
                  , filename = 'invoice.pdf' )
    msg.attach(att)

    if '@' in client_email:
        server.sendmail('airoptions.llc@gmail.com', client_email, msg.as_string())
    # otherwise dont send mail 
    server.close()


#
# main file, presenting everything 
#

def time_now():
    """
    Returns the time right now

    """

    lt = time.localtime()
    return str(lt.tm_year) + '_' + str(lt.tm_mon) + '_' + str(lt.tm_mday) + '_' + \
           str(lt.tm_hour) + '_' + str(lt.tm_min) + '_' + str(lt.tm_sec)


as_of = time_now()
# TODO: FIX HERE !!!
lt_slash = str(lt.tm_mon) + '/' + str(lt.tm_mday) + '/' +  str(lt.tm_year)
lt_str = ds.convert_dateslash_str(lt_slash)
# form 
# form = cgi.FieldStorage()
nonce = form.getvalue('nonce')
return_ow_final = form.getvalue('return_ow_final')
ow_ind = return_ow_final == 'one-way'  # indicator for one-way flights
if ow_ind:
    all_valid, origin_place, dest_place, option_start, option_end, \
        outbound_start, outbound_end, strike, carrier_used, \
        return_ow, cabin_class, nb_people, client_email_addr = get_data_final(form, lt_slash)
else:
    all_valid, origin_place, dest_place, option_start, option_end, \
        outbound_start, outbound_end, strike, carrier_used, \
        option_start_ret, option_end_ret, inbound_start, inbound_end, \
        return_ow, cabin_class, nb_people, client_email_addr = get_data_final(form, lt_slash)

# card data
card_name      = form.getvalue('card-name')
card_address   = form.getvalue('card-address')
sq_postal_code = form.getvalue('sq-postal-code-final')
city_state     = form.getvalue('card-city')

orig_long, dest_long, orig_short, dest_short = process_orig_dest(origin_place, dest_place)

amount_charged = int(form.getvalue('option_value_final')) * 100  # in cents - transforms to dollars
# string of reorg_flights_curr (in string form)

flights_sel = form.getvalue('flights_sel_final')
flights_d = json.loads(flights_sel)  # this is either a list or dict

with open(log_file + '.' + getpass.getuser(), 'a') as eo_log:
    eo_log.write('FLIGHTS\n')
    if type(flights_sel) == list:
        eo_log.write(json.dumps(flights_d[0]))
        eo_log.write(json.dumps(flights_d[1]))
    else:
        eo_log.write(json.dumps(flights_d))
    eo_log.write('FLIGHTS_END\n')


# check if no-machinations took place - recompute the value & determine
if all_valid:

    way_args = { 'origin_place'       : orig_short
               , 'dest_place'         : dest_short
               , 'flights_include'    : flights_d
               , 'option_start_date'  : lt_str
               , 'option_end_date'    : option_end
               , 'outbound_date_start': outbound_start
               , 'outbound_date_end'  : outbound_end
               , 'carrier'            : carrier_used
               , 'cabinclass'         : cabin_class
               , 'adults'             : np.int(nb_people)
               , 'K'                  : np.double(strike)
               , 'price_by_range'     : False }

    if not ow_ind:  # return flight, update the argument
        way_args.update({ 'option_ret_start_date': lt_str
                        , 'option_ret_end_date'  : option_end_ret
                        , 'inbound_date_start'   : inbound_start
                        , 'inbound_date_end'     : inbound_end
                        , 'return_flight'        : True } )

    result, price_range, flights_v, reorg_flights_v, minmax_v = ao.compute_option_val(way_args)

    # process_go_ahead
    result_int = np.int(result['avg'] * 100)  # in cents 
    process_go_ahead = (result_int * ao_codes.amount_charged_below < amount_charged) and all_valid

else: # not all valid

    result_int = -1
    process_go_ahead = False

# another check on process_go_ahead
# checked_inputs = check_inputs(return_ow_final, card_name, sq_postal_code, city_state)
checked_inputs = True
process_go_ahead = process_go_ahead and checked_inputs

    
# The access token to use in all Connect API requests. Use your *sandbox* access
# token if you're just testing things out.
access_token = 'sandbox-sq0atb-dFp-iZvsSxWxvpkyMkmhhQ'
# access_token = 'sq0atp-xwb7PhoYou_Ei4fUMctnjg'  # real token 

# The ID of the business location to associate processed payments with.
# See [Retrieve your business's locations]
# (https://docs.connect.squareup.com/articles/getting-started/#retrievemerchantprofile)
# for an easy way to get your business's location IDs.
# If you're testing things out, use a sandbox location ID.
location_id = 'CBASEA8BYmfgA3bG_tLwUc1gJIQ'  # sandbox id 
# location_id = 'CRCTVPQBM1Y1W'  # real id 
api_instance = TransactionApi()

# Every payment you process with the SDK must have a unique idempotency key.
# If you're unsure whether a particular payment succeeded, you can reattempt
# it with the same idempotency key without worrying about double charging
# the buyer.
idempotency_key = str(uuid.uuid1())

# Monetary amounts are specified in the smallest unit of the applicable currency.
# This amount is in cents. It's also hard-coded for $1.00, which isn't very useful.
amount = { 'amount'  : amount_charged
         , 'currency': 'USD' }
body   = { 'idempotency_key': idempotency_key
         , 'card_nonce'     : nonce
         , 'amount_money'   : amount }

# The SDK throws an exception if a Connect endpoint responds with anything besides
# a 200-level HTTP code. This block catches any exceptions that occur from the request.  
if process_go_ahead:
    try: # Charge
        api_response = api_instance.charge(access_token, location_id, body)
        res = api_response.transaction
    except ApiException as e:
        res = "Exception when calling TransactionApi->charge: {}".format(e)

# writing all files, etc 
add_string = as_of + '_' + str(random_nb) + '/'
inquiry_dir = '/home/brumen/public_html/inquiry/' + add_string  # new folder for invoice
inquiry_logs_dir = ao_codes.inquiry_dir + add_string  # new folder for logs 
if not os.path.exists(inquiry_dir):
    os.makedirs(inquiry_dir)
if not os.path.exists(inquiry_logs_dir):
    os.makedirs(inquiry_logs_dir)

# purchase html file construction 
purchase_html = inquiry_dir + 'purchase_' + as_of + '.html'
purchase_pdf = inquiry_dir + 'purchase_' + as_of + '.pdf'
# invoice file names 
invoice_tex = inquiry_dir + 'invoice_' + as_of + '.tex'
invoice_aux = inquiry_dir + 'invoice_' + as_of + '.aux'
invoice_log = inquiry_dir + 'invoice_' + as_of + '.log'
invoice_pdf = inquiry_dir + 'invoice_' + as_of + '.pdf'
invoice_pdf_loc = "https://airoptions-llc.com/inquiry/" + add_string + "/invoice_" + as_of + ".pdf"

# final writing of the html files, invoice files 
if process_go_ahead:  # write all this 
    purchase_html_file = open(purchase_html, 'w')
    template = open('/home/brumen/public_html/results_page/display_results.html', 'r')
    invoice_fo = open(invoice_tex, 'w')
    invoice_template = open(config.prod_dir + 'invoice/invoice_template.tex', 'r')

    #
    write_file_fct( purchase_html_file
                  , template
                  , orig_long
                  , dest_long
                  , orig_short
                  , dest_short
                  , invoice_pdf_loc
                  , amount_charged
                  , nb_people
                  , cabin_class
                  , return_ow_final
                  , flights_d )

    # write invoice
    invoice_vars = invoice_tex, invoice_aux, invoice_log, invoice_pdf 
    if ow_ind:
        option_end_ret = 'unimp'  # not important, simply that it exists 

    write_invoice_fct( invoice_template
                     , invoice_fo
                     , orig_long
                     , dest_long
                     , orig_short
                     , dest_short
                     , card_name
                     , card_address
                     , city_state
                     , nb_people
                     , cabin_class
                     , sq_postal_code
                     , strike
                     , amount_charged
                     , return_ow_final
                     , option_end
                     , option_end_ret
                     , flights_d
                     , invoice_vars
                     , inquiry_logs_dir )

    send_email_to_client(client_email_addr, invoice_pdf)

    invoice_fo.close()
    invoice_template.close()
    purchase_html_file.close()
    template.close()
    # write webpage results 
    fo_again = open(purchase_html, 'r')
    for l in fo_again:
        print(l)
else:
    print('Content-type:text/html\r\n\r\n\n')
    print('<html>\n')
    print('<head>\n')
    print('<title>Ooops, something went wrong, please try again!</title>\n')
    print('</head>\n')
    print('<body>\n')
    print('<h2>Ooops, something went wrong, please try again!</h2>')
    print('</body>\n')
    print('</html>\n')
