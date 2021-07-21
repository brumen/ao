""" codes used for the modules
"""

import os
import datetime
import ao.ds as ds


# mysql database setup
DB_HOST           = 'localhost'  # 'odroid.local'  # db for service
DB_HOST_CALIBRATE = 'prasic.local'  # db for calibration
DATABASE          = 'ao'
DB_USER           = 'brumen'

# Common properties
COUNTRY  = 'US'
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
hotwire_api_key       = 'vxkjbx6j7jzvpt97grskk5bu'
brumen_mysql_pass     = 'PASSWORD2'
airoptions_gmail_pass = 'PASSWORD1'
airoptions_gmail_acct = 'airoptions.llc@gmail.com'

# times of days
morning = (datetime.time(6, 0, 0), datetime.time(10, 40, 0))
afternoon = (datetime.time(10, 41, 0), datetime.time(18, 0, 0))
evening   = (datetime.time(18, 0, 0), datetime.time(23, 0, 0))
night = (datetime.time(23, 0, 0), datetime.time(6, 0, 0))

day_str    = ('morning', 'afternoon', 'evening', 'night')
summer     = '05-01', '09-30'
winter     = '10-01', '04-30'
season_str = ('summer', 'winter')

weekday_days = [0, 1, 2, 3, 4]
weekend_days = [5, 6]

# livedb prices should only be used if within the livedb_delay of being logged
livedb_delay = datetime.timedelta(hours=1)

# working directory
tmp_dir = 'tmp'
log_dir = 'log'
compute_dir = os.path.join(tmp_dir , 'inquiry/compute'    )
inquiry_dir = os.path.join(tmp_dir , 'inquiry'            )
error_log   = os.path.join(log_dir , 'logger/ao.log'      )
debug_dir   = os.path.join(log_dir , 'debug'              )

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

    if morning[0] < hour_dt < morning[1]:
        return 'morning'

    if afternoon[0] < hour_dt < afternoon[1]:
        return 'afternoon'

    if evening[0] < hour_dt < evening[1]:
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
