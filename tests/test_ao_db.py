""" Test cases for manipulation of the AO database.
"""

import datetime
import datetime as dt

from unittest import TestCase

# TODO: FIX THIS BELOW HERE
import ao.ao_db      as ao_db
import ao.air_option as ao


class TestAoDb(TestCase):
    """ Testing of the database aspects of the AO
    """

    def setUp(self):
        self.today_date_plus_2m = ao.date_today() + dt.timedelta(days=60)

    def test_insert_flight(self):
        """ Tests whether the insert flight executes.
        """

        res = ao_db.insert_flight( 'EWR'
                                 , 'SFO'
                                 , datetime.date(2018, 3, 15)
                                 , includecarriers = 'UA'
                                 , dummy           = True
                                 , depth_max = 2 )

        self.assertTrue(True)

    def test_insert_into_db(self):
        """ Test for insert_into_db function.
        """

        res = ao_db.accumulate_flights( 'EWR'
                                      , 'SFO'
                                      , self.today_date_plus_2m
                                      , includecarriers= ['UA']
                                      , curr_depth = 0
                                      , depth_max  = 2 )

        self.assertTrue(True)
