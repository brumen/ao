# testing function for the search routines of air options.

import unittest
import datetime as dt

from pprint import pprint

import datetime
import air_search as aos


class TestAirSearch(unittest.TestCase):

    def setUp(self):
        self.today_date_plus_2m = datetime.date.today() + dt.timedelta(days=60)

    def test_get_itins(self):
        """
        Checks if the get_itins functions run at all

        """

        res = aos.get_itins(outbound_date = self.today_date_plus_2m)
        pprint(res['Itineraries'])

        self.assertTrue(True)

    def test_get_all_carriers(self):
        """ Tests whether the get_all_carriers function runs.
        """

        res = aos.get_all_carriers( 'EWR'
                                  , 'SFO'
                                  ,  datetime.date.today() + dt.timedelta(days=30) )

        self.assertTrue(True)

    def test_get_ticket_prices(self):
        res1 = aos.get_ticket_prices( 'SIN'
                                    , 'KUL'
                                    , self.today_date_plus_2m)

        res2 = aos.get_ticket_prices( 'EWR'
                                    , 'SFO'
                                    , self.today_date_plus_2m)

        self.assertTrue(True)

    def test_flight_particular(self):
        """ Tests for UA flight between LGA and ATL.
        """

        res1 = aos.get_ticket_prices('LGA', 'ATL', dt.date(2018, 4, 17))

        self.assertTrue(True)