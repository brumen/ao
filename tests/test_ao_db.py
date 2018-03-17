# test cases for the ao_db module

import unittest
import datetime
import datetime as dt

import ao_db
import air_option as ao
import air_search as aos


class TestAoDb(unittest.TestCase):
    """
    Testing of the database aspects of the AO

    """

    def setUp(self):
        self.today_date_plus_2m = ao.date_today() + dt.timedelta(days=60)

    def test_insert_flight(self):
        """
        Tests whether the insert flight executes.

        """

        res = ao_db.insert_flight( 'EWR'
                                 , 'SFO'
                                 , datetime.date(2018, 3, 15)
                                 , includecarriers = 'UA'
                                 , dummy           = True
                                 , depth_max = 2 )

        self.assertTrue(True)

    def test_commit_flights_to_live(self):
        """
        Tests the commit_flights_to_live function.

        """

        # flights_l = (as_of, orig, dest, dep_date, arr_date, carrier, price, outbound_leg_id, flight_nb)
        flights_l = [( datetime.datetime.now()
                    , 'SFO'
                    , 'EWR'
                    , '2018-03-14T22:04:12'
                    , '2018-03-15T08:00:00'
                    , 'UA'
                    , 150.
                    , 'id_here'
                    , 110 )]

        ao_db.commit_flights_to_live(flights_l)

        flights_l_2 = [(datetime.datetime.now()
                      , 'SFO'
                      , 'EWR'
                      , '2018-03-14T22:04:12'
                      , '2018-03-15T08:00:00'
                      , 'UA'
                      , 150.
                      , 'id_here'
                      , 110)
                     , (datetime.datetime.now()
                      , 'SFO'
                      , 'EWR'
                      , '2018-04-14T22:04:12'
                      , '2018-04-15T08:00:00'
                      , 'UA'
                      , 250.
                      , 'id_here'
                      , 110)]

        ao_db.commit_flights_to_live(flights_l_2)

        self.assertTrue(True)

    def test_insert_into_db(self):
        """
        Test for insert_into_db function.

        """

        res = ao_db.accumulate_flights( 'EWR'
                                      , 'SFO'
                                      , self.today_date_plus_2m
                                      , includecarriers= ['UA']
                                      , curr_depth = 0
                                      , depth_max  = 2 )

        self.assertTrue(True)
