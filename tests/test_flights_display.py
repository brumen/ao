# Test case for Air Option Estimate module

import numpy  as np

from unittest import TestCase

from ao_estimate       import find_flight_ids
from ui.flight_display import ArrayButtons, flight_price_get


class TestAoEstimate(TestCase):

    def display_flights( self
                       , orig       = 'SFO'
                       , dest       = 'EWR'
                       , carrier    = 'UA'
                       , min_nb_obs = 0):
        """
        Tests of the find_flight_ids function

        """
        df1 = find_flight_ids( orig
                             , dest
                             , carrier
                             , min_nb_obs = min_nb_obs)

        fids = np.array(df1['flight_id'][0:100]).reshape((10,10))
        ArrayButtons(fids)

    def test_flight_price_get(self):

        res = flight_price_get(218312)
        print ("RES: ", res)

        self.assertTrue(True)
