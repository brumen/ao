# getting data from the webpage
from dateutil.parser import parse

# ao modules 
import ds
from ao_codes import iata_cities_codes, iata_codes_cities, iata_airlines_codes, iata_codes_airlines


def validate_airport(airport):
    """
    extract the code from the airport if code not given
    1. if code given, return code, 
    2. if airport name given, look into db 

    """

    airport_upper = airport.upper()

    if airport_upper in iata_codes_cities.keys():  # airport is in IATA code give
        return airport_upper, True

    else:  # airport has a name
        airport_keys_upper = [x.upper() for x in iata_cities_codes.keys()]
        if airport_upper in airport_keys_upper:
            airport_idx = airport_keys_upper.index(airport_upper)
            airport_name = iata_cities_codes.keys()[airport_idx]
            return iata_cities_codes[airport_name], True  # return the code
        else:
            return "Invalid", False


def validate_airline(airline):
    """
    same as above, just for airlines 

    """

    airline_upper = airline.upper()

    if airline_upper in iata_codes_airlines.keys():
        return airline_upper, True

    else:
        iata_airlines_upper = [x.upper() for x in iata_airlines_codes.keys()]
        if airline_upper in iata_airlines_upper:
            airline_idx = iata_airlines_upper.index(airline_upper)
            airline_name = iata_airlines_codes.keys()[airline_idx]
            return iata_airlines_codes[airline_name], True
        else:
            return "Invalid", False


def validate_outbound_dates(date_):
    """
    validates whether date is in the 1/2/2017 format, and converts it into - format

    """

    try:
        parse(date_)
    except ValueError:
        return "", False
    return ds.convert_dateslash_dash(date_), True  # if everything OK, then convert


def validate_option_dates(date_):
    """
    validates whether date is in the 1/2/2017 format

    """

    try:
        parse(date_)
    except ValueError:
        return "", False
    return ds.convert_dateslash_str(date_), True  # if everything OK, then convert


def validate_strike(strike_i):

    try:
        float(strike_i)
        return float(strike_i), True
    except ValueError:
        return -1., False

    
def get_data(form):
    """
    obtains data from the form and returns them 

    """

    all_valid = True
    origin_place,   origin_valid = validate_airport(form.getvalue("origin_place"))
    dest_place,     dest_valid   = validate_airport(form.getvalue("dest_place"))
    option_start,   os_valid     = validate_option_dates(form.getvalue("option_start"))
    option_end,     oe_valid     = validate_option_dates(form.getvalue("option_end"))
    outbound_start, obs_valid    = validate_outbound_dates(form.getvalue("outbound_start"))  # "10/1/2016"
    outbound_end,   obe_valid    = validate_outbound_dates(form.getvalue("outbound_end"))  #  "10/2/2016"

    # check that outbound_start < outbound_end
    if obs_valid and obe_valid:
       outbound_start_dt = parse(outbound_start)
       outbound_end_dt = parse(outbound_end)
       all_valid = all_valid and (outbound_start_dt <= outbound_end_dt)

    strike, strike_valid = validate_strike(form.getvalue("ticket_price"))
    carrier, carrier_valid = validate_airline(form.getvalue("airline_name"))
    # getting the return flight information
    return_ow = form.getvalue("return_ow")  # this is either 'return' or 'one-way'
    cabin_class = form.getvalue("cabin_class")  # cabin class 
    nb_people = form.getvalue("nb_people")
    all_valid = all_valid and origin_valid and dest_valid and obs_valid and obe_valid and carrier_valid and strike_valid

    if return_ow == 'return':
        option_start_ret, ors_valid = validate_option_dates(form.getvalue("option_ret_start"))
        option_end_ret, ore_valid = validate_option_dates(form.getvalue("option_ret_end"))
        inbound_start, ibs_valid = validate_outbound_dates(form.getvalue("outbound_start_ret"))  # "10/1/2016"
        inbound_end, ibe_valid = validate_outbound_dates(form.getvalue("outbound_end_ret"))  #  "10/2/2016"
        # check that inbound_start < inbound_end
        if ibs_valid and ibe_valid:
            inbound_start_dt = parse(inbound_start)
            inbound_end_dt = parse(inbound_end)
            all_valid = all_valid and (inbound_start_dt <= inbound_end_dt)
            # check that inbound dates are after the outbound dates 
            all_valid = all_valid and (parse(outbound_end) < inbound_start_dt)
            
        all_valid = all_valid and ibs_valid and ibe_valid
        
    if carrier == "":
        carrier_used = None
    else:
        carrier_used = carrier

    if return_ow == 'one-way':
        return (all_valid, origin_place, dest_place, option_start, option_end,
                outbound_start, outbound_end, strike, carrier_used,
                return_ow, cabin_class, nb_people)
    else:  # return flight 
        return (all_valid, origin_place, dest_place, option_start, option_end,
                outbound_start, outbound_end, strike, carrier_used,
                option_start_ret, option_end_ret, inbound_start, inbound_end,
                return_ow, cabin_class, nb_people)


def get_data_final(form, start_date):
    """
    obtains data from the form (from form _final for booking) and returns them 
    """
    all_valid = True 
    origin_place, origin_valid = validate_airport(form.getvalue("origin_final"))
    dest_place, dest_valid = validate_airport(form.getvalue("dest_final"))
    option_start, os_valid = start_date, True  # '1/1/2017 ??? 
    option_end, oe_valid = validate_option_dates(form.getvalue('opt_end_dep_final'))  # '2/3/2016
    outbound_start, obs_valid = validate_outbound_dates(form.getvalue("dep_start_final"))
    outbound_end, obe_valid = validate_outbound_dates(form.getvalue("dep_end_final"))

    # check that outbound_start < outbound_end
    if obs_valid and obe_valid:
       outbound_start_dt = parse(outbound_start)
       outbound_end_dt = parse(outbound_end)
       all_valid = all_valid and (outbound_start_dt <= outbound_end_dt)

    strike, strike_valid = validate_strike(form.getvalue("ticket_price_final"))
    carrier, carrier_valid = validate_airline(form.getvalue("carrier_final"))
    # getting the return flight information
    return_ow = form.getvalue("return_ow_final")  # this is either 'return' or 'one-way'
    cabin_class = form.getvalue("class_travel")  # cabin class 
    nb_people = form.getvalue("nb_persons")
    client_email_addr = form.getvalue("email-addr")
    all_valid = all_valid and origin_valid and dest_valid and obs_valid and obe_valid and carrier_valid and strike_valid
    
    if return_ow == 'return':
        option_start_ret, ors_valid = start_date, True  # '2017-03-01'
        option_end_ret, ore_valid = validate_option_dates(form.getvalue('opt_end_ret_final'))  # '2017-04-01', True
        inbound_start, ibs_valid = validate_outbound_dates(form.getvalue("ret_start_final"))
        inbound_end, ibe_valid = validate_outbound_dates(form.getvalue("ret_end_final"))
        # check that inbound_start < inbound_end
        if ibs_valid and ibe_valid:
            inbound_start_dt = parse(inbound_start)
            inbound_end_dt = parse(inbound_end)
            all_valid = all_valid and (inbound_start_dt <= inbound_end_dt)
            # check that inbound dates are after the outbound dates 
            all_valid = all_valid and (parse(outbound_end) < inbound_start_dt)
            
        all_valid = all_valid and ibs_valid and ibe_valid
        
    if carrier == "":
        carrier_used = None
    else:
        carrier_used = carrier

    if return_ow == 'one-way':
        return (all_valid, origin_place, dest_place, option_start, option_end,
                outbound_start, outbound_end, strike, carrier_used,
                return_ow, cabin_class, nb_people, client_email_addr)
    else:  # return flight 
        return (all_valid, origin_place, dest_place, option_start, option_end,
                outbound_start, outbound_end, strike, carrier_used,
                option_start_ret, option_end_ret, inbound_start, inbound_end,
                return_ow, cabin_class, nb_people, client_email_addr)


def get_data_dict(form):
    """
    obtains data from the form in dict format (for POST) and returns them 

    :param form: form of the data passed from apache
    :type form:  TODO: FINISH HERE. 
    """
    all_valid = True
    origin_place,   origin_valid  = validate_airport(form["origin_place"])
    dest_place,     dest_valid    = validate_airport(form["dest_place"])
    option_start,   os_valid      = validate_option_dates(form["option_start"])
    option_end,     oe_valid      = validate_option_dates(form["option_end"])
    outbound_start, obs_valid     = validate_outbound_dates(form["outbound_start"])  # "10/1/2016"
    outbound_end,   obe_valid     = validate_outbound_dates(form["outbound_end"])  #  "10/2/2016"
    strike,         strike_valid  = validate_strike(form["ticket_price"])
    carrier,        carrier_valid = validate_airline(form["airline_name"])
    if obs_valid and obe_valid:
        outbound_start_dt = parse(outbound_start)
        outbound_end_dt   = parse(outbound_end)
        all_valid         = all_valid and (outbound_start_dt <= outbound_end_dt)

    # getting the return flight information
    return_ow   = form["return_ow"]  # this is either 'return' or 'one-way'
    cabin_class = form["cabin_class"]  # cabin class
    nb_people   = form["nb_people"]
    all_valid   = all_valid and origin_valid and dest_valid and obs_valid and obe_valid and carrier_valid and strike_valid

    if return_ow == 'return':
        option_start_ret, osr_valid = validate_option_dates(form["option_ret_start"])
        option_end_ret,   oer_valid = validate_option_dates(form["option_ret_end"])
        inbound_start,    ibs_valid = validate_outbound_dates(form["outbound_start_ret"])  # "10/1/2016"
        inbound_end,      ibe_valid = validate_outbound_dates(form["outbound_end_ret"])  #  "10/2/2016"
        if ibs_valid and ibe_valid:
            inbound_start_dt = parse(inbound_start)
            inbound_end_dt   = parse(inbound_end)
            all_valid        = all_valid and (inbound_start_dt <= inbound_end_dt)
            # check that inbound dates are after the outbound dates 
            all_valid        = all_valid and (parse(outbound_end) < inbound_start_dt)
        
        all_valid = all_valid and ibs_valid and ibe_valid
        
    if carrier == "":
        carrier_used = None
    else:
        carrier_used = carrier

    if return_ow == 'one-way':
        return (all_valid, origin_place, dest_place, option_start, option_end,
                outbound_start, outbound_end, strike, carrier_used,
                return_ow, cabin_class, nb_people)
    else:  # return flight 
        return (all_valid, origin_place, dest_place, option_start, option_end,
                outbound_start, outbound_end, strike, carrier_used,
                option_start_ret, option_end_ret, inbound_start, inbound_end,
                return_ow, cabin_class, nb_people)
