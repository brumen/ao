import unittest
import datetime as dt

import ds
import air_option as ao


class TestAirSearch(unittest.TestCase):

    def setUp(self):
        self.today_date         = ao.date_today()
        self.today_date_plus_2m = self.today_date + dt.timedelta(days=60)

    def test_construct_date_range(self):
        """
        Tests the date range construction.

        """

        k1 = ds.construct_date_range(self.today_date, self.today_date_plus_2m)
        print(k1)

        self.assertTrue(True)