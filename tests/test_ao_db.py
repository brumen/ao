# test cases for the ao_db module

import unittest
import datetime

import ao_db


class TestAoDb(unittest.TestCase):
    """
    Testing of the database aspects of the AO

    """

    def test_insert_flight(self):
        """
        Tests whether the insert flight executes.

        """

        res = ao_db.insert_flight( 'EWR'
                                 , 'SFO'
                                 , datetime.date(2018, 3, 15)
                                 , includecarriers = 'UA'
                                 , dummy           = True
                                 , depth_max = 2 )

        self.assertTrue(True)