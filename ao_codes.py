# codes used for the modules 
import config
import datetime as dt
import ds 
# import csv
import os

# mysql database setup
DB_HOST           = 'odroid.local'  # db for service
DB_HOST_CALIBRATE = 'prasic.local'  # db for calibration
DATABASE          = 'ao'
DB_USER           = 'brumen'

# original sqlite db
SQLITE_FILE       = os.path.join(config.work_dir, 'ao.db')
SQLITE_FILE_CLONE = SQLITE_FILE + '.clone'

# Common properties
COUNTRY = 'US'
CURRENCY = 'USD'
LOCALE   = 'en-US'

# maximum flight ticket, used for reordering a list
MAX_TICKET = 1000000.  # one million
MIN_PRICE  = 50.       # minimum price of an air option

# large drifts
LARGE_DRIFT = 500.

# DAY COUNT FACTOR
DCF = 365.

# api keys
skyscanner_api_key    = 'pe941949487693197945430744449137'
hotwire_api_key       = "vxkjbx6j7jzvpt97grskk5bu"
brumen_mysql_pass     = 'c2D779Mu'
airoptions_gmail_pass = 'MorningMadnessIsHere'
airoptions_gmail_acct = 'airoptions.llc@gmail.com'

# times of days 
# morning = '06:00:00', '11:00:00'
morning   = '06:00:00', '10:40:00'
# afternoon = '11:00:00', '18:00:00'
afternoon = '10:41:00', '18:00:00'
evening   = '18:00:00', '23:00:00'
night     = '23:00:00', '06:00:00'


day_str    = ('morning', 'afternoon', 'evening', 'night')
summer     = '05-01', '09-30'
winter     = '10-01', '04-30'
season_str = ('summer', 'winter')

# derived
morning_dt   = ds.convert_hour_time(morning[0])  , ds.convert_hour_time(morning[1]  )
afternoon_dt = ds.convert_hour_time(afternoon[0]), ds.convert_hour_time(afternoon[1])
evening_dt   = ds.convert_hour_time(evening[0])  , ds.convert_hour_time(evening[1]  )
night_dt     = ds.convert_hour_time(night[0])    , ds.convert_hour_time(night[1]    )

weekday_days = [0, 1, 2, 3, 4]
weekend_days = [5, 6]

# livedb prices should only be used if within the livedb_delay of being logged
livedb_delay = dt.timedelta(hours=1)

# working directory
compute_dir = os.path.join(config.tmp_dir , 'inquiry/compute'    )
inquiry_dir = os.path.join(config.tmp_dir , 'inquiry'            )
error_log   = os.path.join(config.log_dir , 'logger/ao.log'      )
debug_dir   = os.path.join(config.log_dir , 'debug'              )
#iata_dir    = os.path.join(config.prod_dir, 'iata'               )
#iata_file   = os.path.join(iata_dir       , 'iata_codes_work.csv')

# reserves, taxes
reserves             = 0.12  # 10% reserves, 15, but
tax_rate             = 0.09  # slightly more than 8%
ref_base_F           = 600.  # 50$ is based on reference of 600$
amount_charged_below = 0.95


def get_tod(time_str):
    """
    gets the time of day if given time string

    :param time_str:   TODO: time_str should be some time, not a string
    """

    hour_dt = ds.convert_hour_time(time_str)

    if morning_dt[0] < hour_dt < morning_dt[1]:
        return 'morning'

    if afternoon_dt[0] < hour_dt < afternoon_dt[1]:
        return 'afternoon'

    if evening_dt[0] < hour_dt < evening_dt[1]:
        return 'evening'

    return 'night'


def get_weekday_ind(week_day_int : int) -> str:
    """
    returns whether the week_day in integer format is a week day or a weekend day

    :param week_day_int: integer value of the week day
    :returns:            weekday, weekend
    """

    return 'weekday' if week_day_int in weekday_days else 'weekend'


# clusters of airports that can be considered together
clusters = { 'NYCA': ('JFK', 'EWR', 'LGA'),
             'HOUA': ('HOU', 'IAH') }
# ticket fees
fees = {"Airtran"  : {"ticket_change": 150.,
                      "same day": 50.},
        "Alaska"   : {"ticket_change": 125.,
                      "same day": 25.},
        "Allegiant": {"ticket_change": 50.},
        "American" : {"ticket_change": 200.,
                      "same day": 75.},
        "Delta":     {"ticket_change": 200.,
                      "same day": 50.},
        "Frontier":  {"ticket_change": 50.},
        "JetBlue":   {"ticket_change": 75.,
                      "same day": 50.},
        "Hawaiian":  {"ticket_change": 30.}
        }


# TODO: DELETE THIS PART - partially already in iata/codes.py
# def import_iata_codes():
#     """
#     Import iata codes from the file iata_file
#
#     """
#
#     iata_cities_codes = {city: code
#                          for code, country, city in csv.reader( open(iata_file, 'r', encoding='utf-8')
#                                                               , delimiter=',')}
#
#     iata_codes_cities = {code: city
#                          for code, country, city in csv.reader( open(iata_file, 'r', encoding='utf-8')
#                                                               , delimiter=',')}
#
#     # read iata airlines
#     # airline, code, three_digit, icao, country = iata airlines stuff
#     iata_codes_airlines = { code: airline
#                             for airline, code, three_digit, icao, country in
#                                 csv.reader( open(os.path.join(iata_dir, 'iata_airlines.csv'), 'r', encoding='utf-8')
#                                           , delimiter=',') }
#
#     iata_airlines_codes = { airline: code
#                             for airline, code, _, _, _ in
#                                 csv.reader( open(os.path.join(iata_dir, 'iata_airlines.csv'), 'r', encoding='utf-8')
#                                           , delimiter=',') }
#
#     return iata_cities_codes, iata_codes_cities, iata_airlines_codes, iata_codes_airlines
#
#
# iata_cities_codes, iata_codes_cities, iata_airlines_codes, iata_codes_airlines = import_iata_codes()
