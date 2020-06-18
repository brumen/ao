import unittest
import datetime

from ao.ds import construct_date_range


class TestAirSearch(unittest.TestCase):

    def setUp(self):
        self.today_date         = datetime.date.today()
        self.today_date_plus_2m = self.today_date + datetime.timedelta(days=60)

    def test_construct_date_range(self):
        """ Tests the date range construction.
        """

        date_range = construct_date_range(self.today_date, self.today_date_plus_2m)

        self.assertTrue(isinstance(date_range, list))
