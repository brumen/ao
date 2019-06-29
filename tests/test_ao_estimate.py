# Test case for Air Option Estimate module
import unittest
import pandas as pd
import numpy  as np


import ao_db
import ao_estimate as aoe
from ao_estimate import find_flight_ids, ArrayButtons


class TestAoEstimate(unittest.TestCase):

    def test_only_1(self):
        """ Tests TODO: HERE

        """

        direct_flights = """
        SELECT DISTINCT as_of, orig, dest, dep_date, arr_date, price 
        FROM flights WHERE orig= 'SFO' AND dest = 'EWR' AND carrier='UA' 
        ORDER BY as_of"""

        dep_dates_str = """
        SELECT DISTINCT dep_date FROM flights WHERE orig= 'SFO' AND dest = 'EWR' AND carrier='UA'
        """
        dep_dates = pd.read_sql_query( dep_dates_str
                                     , ao_db.conn_ao
                                     , parse_dates = {'dep_date': '%Y-%m-%dT%H:%M:%S'})

        df1 = pd.read_sql_query( direct_flights
                               , ao_db.conn_ao
                               , parse_dates ={'as_of'   : '%Y-%m-%d',
                                               'dep_date': '%Y-%m-%dT%H:%M:%S',
                                               'arr_date': '%Y-%m-%dT%H:%M:%S'})

        return df1, dep_dates

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

        res = aoe.flight_price_get(218312)
        print ("RES: ", res)

        self.assertTrue(True)

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

        print (aoe.flight_corr(orig_l, dest_l, carrier_l, dep_date_l))

    def test_flight_vol_mysql(self):
        """
        Tests the
        """
        orig = 'SFO'
        dest = 'EWR'
        carrier = 'UA'

        print (aoe.flight_vol_mysql(orig, dest, carrier))

        self.assertTrue(True)
