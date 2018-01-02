# codes used for the modules 
import config
import datetime as dt
import ds 
import csv

# api keys
skyscanner_api_key    = 'pe941949487693197945430744449137'
hotwire_api_key       = "vxkjbx6j7jzvpt97grskk5bu"
brumen_mysql_pass     = 'c2D779Mu'
airoptions_gmail_pass = 'MorningMadnessIsHere'
airoptions_gmail_acct = 'airoptions.llc@gmail.com'

# times of days 
# morning = '06:00:00', '11:00:00'
morning    = '06:00:00', '10:40:00'
# afternoon = '11:00:00', '18:00:00'
afternoon = '10:41:00', '18:00:00'
evening = '18:00:00', '23:00:00'
night = '23:00:00', '06:00:00'


day_str    = ('morning', 'afternoon', 'evening', 'night')
summer     = '05-01', '09-30'
winter     = '10-01', '04-30'
season_str = ('summer', 'winter')

# derived
morning_dt   = ds.convert_hour_time(morning[0]), ds.convert_hour_time(morning[1])
afternoon_dt = ds.convert_hour_time(afternoon[0]), ds.convert_hour_time(afternoon[1])
evening_dt   = ds.convert_hour_time(evening[0]), ds.convert_hour_time(evening[1])
night_dt     = ds.convert_hour_time(night[0]), ds.convert_hour_time(night[1])

weekday_days = [0, 1, 2, 3, 4]
weekend_days = [5, 6]

livedb_delay = dt.timedelta(hours=1)

# working directory
compute_dir = config.prod_dir + 'inquiry/compute/'
inquiry_dir = config.prod_dir + 'inquiry/'
error_log   = config.prod_dir + 'logger/ao.log'
debug_dir   = config.prod_dir + 'debug/'
iata_dir    = config.prod_dir + 'iata/'

# reserves, taxes
reserves = 0.12  # 10% reserves, 15, but 
tax_rate = 0.09  # slightly more than 8%
ref_base_F = 600.  # 50$ is based on reference of 600$    
amount_charged_below = 0.95


def get_tod(time_str):
    """
    gets the tod if given time string 

    :param time_str:   TODO: ????
    """
    hour_dt = ds.convert_hour_time(time_str)
    morning_ind = hour_dt > morning_dt[0] and hour_dt < morning_dt[1]
    afternoon_ind = hour_dt > afternoon_dt[0] and hour_dt < afternoon_dt[1]
    evening_ind = hour_dt > evening_dt[0] and hour_dt < evening_dt[1]
    night_ind = hour_dt > night_dt[0] and hour_dt < night_dt[1]
    if morning_ind:
        time_of_day_res = 'morning'
    elif afternoon_ind:
        time_of_day_res = 'afternoon'
    elif evening_ind:
        time_of_day_res = 'evening'
    else:
        time_of_day_res = 'night'
        
    return time_of_day_res 


def get_weekday_ind(week_day_int):
    """
    returns whether the week_day in integer format is a week day or a weekend day

    """

    if week_day_int in weekday_days:
        return 'weekday'
    else:
        return 'weekend'


# clusters of airports that can be considered together
clusters = {'NYCA': ('JFK', 'EWR', 'LGA'),
            'HOUA': ('HOU', 'IAH')}
# ticket fees
fees = {"Airtran":   {"ticket_change": 150.,
                      "same day": 50.},
        "Alaska":    {"ticket_change": 125.,
                      "same day": 25.},
        "Allegiant": {"ticket_change": 50.},
        "American":  {"ticket_change": 200.,
                      "same day": 75.},
        "Delta":     {"ticket_change": 200.,
                      "same day": 50.},
        "Frontier":  {"ticket_change": 50.},
        "JetBlue":   {"ticket_change": 75.,
                      "same day": 50.},
        "Hawaiian":  {"ticket_change": 30.}  # WIDE RANGE HERE CHANGE
        }


def import_iata_codes(iata_file=iata_dir + 'iata_codes_work.csv'):
    """
    import iata codes from the file specified
    """
    iata_reader = csv.reader(open(iata_file, 'r'), delimiter=',')
    iata_cities_codes = {city: code for code, country, city in iata_reader}
    iata_reader = csv.reader(open(iata_file, 'r'), delimiter=',')
    iata_codes_cities = {code: city for code, country, city in iata_reader}
    # read iata airlines
    iata_airlines_reader = csv.reader(open(iata_dir + 'iata_airlines.csv', 'r'),
                                      delimiter=',')
    iata_codes_airlines = {code: airline for airline, code, three_digit, icao, country in
                           iata_airlines_reader}
    iata_airlines_reader = csv.reader(open(iata_dir + 'iata_airlines.csv', 'r'),
                                      delimiter=',')
    iata_airlines_codes = {airline: code for airline, code, three_digit, icao, country in
                           iata_airlines_reader}

    return iata_cities_codes, iata_codes_cities, iata_airlines_codes, iata_codes_airlines


iata_cities_codes, iata_codes_cities, iata_airlines_codes, iata_codes_airlines = import_iata_codes()
