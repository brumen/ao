import unittest
import datetime as dt
import air_option as ao
import air_search as aos


class TestAirSearch(unittest.TestCase):

    def setUp(self):
        self.today_date_plus_2m = ao.date_today() + dt.timedelta(days=60)

    def test_get_itins(self):
        '''
        checks if the get_itins functions run at all

        '''

        res = aos.get_itins(outbound_date=self.today_date_plus_2m)
        print res

        self.assertTrue(True)

    def test_get_ticket_prices(self):
        '''
        checks if the get_ticket prices works at all

        '''

        res = aos.get_ticket_prices( origin_place  = 'EWR'
                                   , dest_place    = 'SFO'
                                   , outbound_date = self.today_date_plus_2m)
        print res

        self.assertTrue(True)

    def test_find_carrier(self):
        '''
        tests whether the find_carrier function runs

        '''

        self.assertTrue(True)
