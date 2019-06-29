import unittest
import datetime as dt

import ds
import datetime


class TestAirSearch(unittest.TestCase):

    def setUp(self):
        self.today_date         = datetime.date.today()
        self.today_date_plus_2m = self.today_date + dt.timedelta(days=60)

    def test_construct_date_range(self):
        """
        Tests the date range construction.

        """

        date_range = ds.construct_date_range(self.today_date, self.today_date_plus_2m)

        self.assertTrue(isinstance(date_range, list))
