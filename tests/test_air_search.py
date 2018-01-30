import unittest
import datetime as dt

import ds
import air_option as ao
import air_search as aos


class TestAirSearch(unittest.TestCase):

    def setUp(self):
        self.today_date_plus_2m = ao.date_today() + dt.timedelta(days=60)

    def test_get_itins(self):
        """
        checks if the get_itins functions run at all

        """

        res = aos.get_itins(outbound_date=self.today_date_plus_2m)
        print res

        self.assertTrue(True)

    def test_find_carrier(self):
        """
        tests whether the find_carrier function runs

        """

        self.assertTrue(True)

    def test_get_all_carriers(self):
        """
        Tests whether the get_all_carriers function runs.

        """
        future_date = ds.convert_date_datedash(ao.date_today() + dt.timedelta(days=30))
        print aos.get_all_carriers( 'EWR'
                                  , 'SFO'
                                  , future_date )

        self.assertTrue(True)

    def test_get_ticket_prices(self):
        res1 = aos.get_ticket_prices( 'SIN'
                                    , 'KUL'  # possible NYCA-sky
                                    , ds.convert_date_datedash(self.today_date_plus_2m))

        res2 = aos.get_ticket_prices( 'EWR'
                                    , 'SFO'
                                    , self.today_date_plus_2m)

        print "RES1: ", res1, res2

        self.assertTrue(True)