# testing function for the search routines of air options.

import datetime as dt
import datetime

from unittest import TestCase

from ao.air_option_derive import AirOptionSkyScanner


class TestAirSearch(TestCase):

    DATE_TODAY = datetime.date(2017, 3, 1)

    def test_get_itins(self):
        """ Checks if the get_itins functions run at all
        """

        res = AirOptionSkyScanner.get_itins('EWR', 'SFO', outbound_date = self.DATE_TODAY)

        self.assertTrue(True)

    def test_get_ticket_prices(self):
        """ TODO
        """

        res1 = AirOptionSkyScanner.get_ticket_prices( 'SIN', 'KUL', self.DATE_TODAY)
        res2 = AirOptionSkyScanner.get_ticket_prices( 'EWR', 'SFO', self.DATE_TODAY)

        self.assertTrue(True)

    def test_flight_particular(self):
        """ Tests for UA flight between LGA and ATL.
        """

        res1 = AirOptionSkyScanner.get_ticket_prices('LGA', 'ATL', dt.date(2018, 4, 17))

        self.assertTrue(True)
