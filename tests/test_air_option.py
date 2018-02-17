# testing framework for air options

import numpy    as np
import datetime as dt
import unittest


import ds
import air_option  as ao
import ao_estimate as aoe


class TestAirOption(unittest.TestCase):

    def setUp(self):
        """
        Setting of the basic variables.

        """

        self.outDate            = ao.date_today() + dt.timedelta( days = 30 )
        self.outDatePlusOne     = self.outDate + dt.timedelta( days = 1 )
        self.optionOutDateStart = ao.date_today() + dt.timedelta ( days = 1 )
        self.optionOutDateEnd   = self.outDate - dt.timedelta ( days = 1 )

        self.retDate            = self.outDate + dt.timedelta ( days = 7 )
        self.retDatePlusOne     = self.retDate + dt.timedelta ( days = 1 )
        self.optionRetDateStart = ao.date_today() + dt.timedelta ( days = 1 )
        self.optionRetDateEnd   = self.outDate - dt.timedelta ( days = 1 )

        self.io_dr = ds.construct_date_range( self.outDate
                                            , self.outDatePlusOne)

    def test_ao_2( self
                 , nb_sim = 50000):
        """
        tests the air option

        """

        tickets = np.linspace(450., 400., 50)
        s_v = 100. * np.ones(len(tickets))
        T_l = np.linspace(1./365., 62./365., 62)
        res = ao.air_option( tickets
                           , s_v
                           , s_v
                           , T_l
                           , T_l + 0.01
                           , 0.2
                           , nb_sim = nb_sim)

        self.assertTrue(True)

    def test_construct_date_range(self):
        """
        Tests whether the date range function works.

        """

        self.assertTrue(self.io_dr == [ self.outDate
                                      , self.outDatePlusOne ])

    def test_obtain_flights(self):
        """
        Tests whether we can obtain flights from SkyScanner

        """

        res = ao.obtain_flights( 'EWR'
                               , 'SFO'
                               , 'UA'
                               , self.io_dr
                               , None)

        self.assertTrue(True)

    def test_ao_25(self):
        """
        tests the air option

        """

        nb_sim = 50000
        tickets = np.linspace(450., 400., 50)
        s_v = 100. * np.ones(len(tickets))
        T_l = np.linspace(1./365., 180./365., 180)

        res = ao.air_option( tickets
                           , s_v
                           , s_v
                           , T_l
                           , 450.
                           , 100.
                           , 0.2
                           , nb_sim = nb_sim)

        self.assertTrue(True)

    def test_ao_26(self, nb_sim=10000):
        """
        tests the air option
        """
        tickets = np.linspace(450., 400., 50)
        s_v = 100. * np.ones(len(tickets))
        T_l = np.linspace(1./365., 180./365., 180)
        T_mat = T_l + 0.01  # some increase over T_l

        res = ao.air_option( tickets
                           , s_v
                           , s_v
                           , T_l
                           , T_mat
                           , 100.
                           , ao_f   = ao.ao_f
                           , nb_sim = nb_sim)

        self.assertTrue(True)

    def test_ao_27( self
                  , nb_sim=10000):
        """
        tests the air option

        """

        tickets = np.linspace(450., 400., 3)
        s_v = 100. * np.ones(len(tickets))
        T_l = np.linspace(1./365., 180./365., 2)
        d_v = 0.2 * np.ones(len(tickets))

        res = ao.air_option( tickets
                           , s_v
                           , d_v
                           , T_l
                           , T_l + 0.05
                           , 450.
                           , ao_f   = ao.ao_f_arb
                           , nb_sim = nb_sim)

        self.assertTrue(True)

    def test_compute_vols( self
                         , airline = 'Alaska Airlines' ):
        """
        airline from the cache database  # TODO: THIS FUNCTION DOES NOT WORK

        """

        print aoe.all_vols_by_airline(airline, use_cache=True)
        self.assertTrue(True)

    def test_simple(self):
        """
        Tests the compute_option_val function

        """

        v1 = ao.compute_option_val( origin_place          = 'SFO'
                                  , dest_place            = 'EWR'
                                  , option_start_date     = self.optionOutDateStart
                                  , option_end_date       = self.optionOutDateEnd
                                  , option_ret_start_date = self.optionRetDateStart
                                  , option_ret_end_date   = self.optionRetDateEnd
                                  , outbound_date_start   = self.outDate
                                  , outbound_date_end     = self.outDatePlusOne
                                  , inbound_date_start    = self.retDate
                                  , inbound_date_end      = self.retDatePlusOne
                                  , K                     = 200.0
                                  , carrier               = 'UA'
                                  , nb_sim                = 10000
                                  , rho                   = 0.95
                                  , adults                = 1
                                  , cuda_ind              = False
                                  , errors                = 'graceful'
                                  , simplify_compute      = 'take_last_only'
                                  , underlyer             = 'n'
                                  , price_by_range        = False
                                  , return_flight         = True )
        print v1
        self. assertTrue(True)

    def test_ao_new2( self
                    , simplify_compute = 'take_last_only'
                    , cuda_ind         = False ):

        v1 = ao.compute_option_val( option_start_date   = self.optionOutDateStart
                                  , option_end_date     = self.optionOutDateEnd
                                  , outbound_date_start = self.outDate
                                  , outbound_date_end   = self.outDatePlusOne
                                  , K                   = 150.0
                                  , nb_sim              = 30000
                                  , rho                 = 0.95
                                  , simplify_compute    = simplify_compute
                                  , return_flight       = False
                                  , cuda_ind            = cuda_ind )

        self.assertTrue(True)

    def test_ao_rho(self):
        """
        tests the effect of correlation

        """
        rho_l      = [0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.2, -0.2, -0.3, -0.9]
        K          = 1000.
        impact_rho = {}
        for rho in rho_l:
            k1 = ao.compute_option_val( outbound_date_start = self.outDate
                                      , outbound_date_end   = self.outDatePlusOne
                                      , option_start_date   = self.optionOutDateStart
                                      , option_end_date     = self.optionOutDateEnd
                                      , inbound_date_start  = self.retDate
                                      , inbound_date_end    = self.retDatePlusOne
                                      , option_ret_start_date = self.optionRetDateStart
                                      , option_ret_end_date   = self.optionOutDateEnd
                                      , K                   = K
                                      , return_flight       = True
                                      , rho                 = rho )
            impact_rho[rho] = k1[1]

        self.assertTrue(True)

    def test_get_flight_data(self):
        """
        Tests whether the get_flight_data function even executes

        """

        res = ao.get_flight_data( outbound_date_start = self.outDate
                                , outbound_date_end   = self.outDatePlusOne
                                , inbound_date_start  = self.retDate
                                , inbound_date_end    = self.retDatePlusOne )

        self.assertTrue(True)
