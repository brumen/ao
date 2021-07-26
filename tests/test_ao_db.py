""" Test cases for manipulation of the AO database.
"""

import datetime

from unittest import TestCase

# TODO: FIX THIS BELOW HERE
from ao.ao_db import insert_flight, accumulate_flights


class TestAoDb(TestCase):
    """ Testing of the database aspects of the AO
    """

    DATE_TODAY = datetime.date(2017, 3, 1)

    def test_insert_flight(self):
        """ Tests whether the insert flight executes.
        """

        res = insert_flight( 'EWR'
                           , 'SFO'
                           , datetime.date(2018, 3, 15)
                           , includecarriers = 'UA'
                           , dummy           = True
                           , depth_max = 2 )

        self.assertTrue(True)

    def test_insert_into_db(self):
        """ Test for insert_into_db function.
        """

        res = accumulate_flights( 'EWR'
                                , 'SFO'
                                , self.DATE_TODAY
                                , includecarriers= ['UA']
                                , curr_depth = 0
                                , depth_max  = 2 )

        self.assertTrue(True)
