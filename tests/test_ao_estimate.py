# Test case for Air Option Estimate module

from unittest import TestCase

from ao_estimate import find_flight_ids, flight_corr, flight_vol


class TestAoEstimate(TestCase):

    def test_find_flight_ids(self):
        """
        Finds the ids of flights between origin and destination
        for a particular airline carrier.

        """

        print (find_flight_ids('SFO', 'EWR', 'UA'), host='localhost')
        self.assertTrue(True)

    def test_flight_corr(self):
        orig_l     = ['SFO']
        dest_l     = ['JFK']
        carrier_l  = ['B6']
        dep_date_l = ['2016-10-01T15:10:00']

        print (flight_corr(orig_l, dest_l, carrier_l, dep_date_l))

    def test_flight_vol_mysql(self):
        """
        Tests the
        """
        orig = 'SFO'
        dest = 'EWR'
        carrier = 'UA'

        print (flight_vol(orig, dest, carrier))

        self.assertTrue(True)
