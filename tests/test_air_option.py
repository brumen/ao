# testing framework for air options

import numpy    as np
import datetime as dt
import unittest


import ds
import air_option  as ao
import ao_estimate as aoe


class TestAirOption(unittest.TestCase):

    def test_ao_2( self
                 , nb_sim=50000):
        """
        tests the air option

        """
        tickets = np.linspace(450., 400., 50)
        s_v = 100. * np.ones(len(tickets))
        T_l = np.linspace(1./365., 62./365., 62)
        print ao.air_option( tickets
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

        io_dr_minus = ao.construct_date_range( '2018-03-01'
                                             , '2018-03-02')

        self.assertTrue(io_dr_minus == ['2018-03-01', '2018-03-02'])

    def test_obtain_flights(self):
        """
        Tests whether we can obtain flights from SkyScanner

        """
        io_dr_minus = ao.construct_date_range( '2018-03-01'
                                             , '2018-03-02')
        res = ao.obtain_flights( 'EWR'
                               , 'SFO'
                               , 'UA'
                               , io_dr_minus
                               , None)
        print res

        self.assertTrue(True)

    def test_ao_25(self):
        """
        tests the air option

        """

        nb_sim = 50000
        tickets = np.linspace(450., 400., 50)
        s_v = 100. * np.ones(len(tickets))
        T_l = np.linspace(1./365., 180./365., 180)

        print ao.air_option( tickets
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

        print ao.air_option( tickets
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

        print ao.air_option( tickets
                           , s_v
                           , d_v
                           , T_l
                           , T_l + 0.05
                           , 450.
                           , ao_f   = ao.ao_f_arb
                           , nb_sim = nb_sim)

        self.assertTrue(True)

    def test_compute_vols( self
                         ,  airline='Alaska Airlines'):
        """
        airline from the cache database
        """

        print aoe.all_vols_by_airline(airline, use_cache=True)
        self.assertTrue(True)

    def test_simple(self):
        """
        Tests the compute_option_val function

        """

        v1 = ao.compute_option_val( origin_place          = 'SFO'
                                  , dest_place            = 'EWR'
                                  , option_start_date     = '20180301'
                                  , option_end_date       = '20180301'
                                  , option_ret_start_date = '20180401'
                                  , option_ret_end_date   = '20180402'
                                  , outbound_date_start   = '2018-04-01'
                                  , outbound_date_end     = '2018-04-01'
                                  , inbound_date_start    = '2018-05-12'
                                  , inbound_date_end      = '2018-05-13'
                                  , K                     = 200.0
                                  , carrier               = 'UA'
                                  , nb_sim                = 10000
                                  , rho                   = 0.95
                                  , adults                = 1
                                  , cuda_ind              = False
                                  , errors                = 'graceful'
                                  , simplify_compute      = 'take_last_only'
                                  , underlyer             = 'ln'
                                  , mt_ind                = True
                                  , price_by_range        = False
                                  , return_flight         = True)

        print v1
        self. assertTrue(True)

    def test_ao_new1( self
                    , simplify_compute = 'take_last_only'):

        v1 = ao.compute_option_val( option_start_date     = '20161211'
                                  , option_end_date       = '20161212'
                                  , option_ret_start_date = '20161201'
                                  , option_ret_end_date   = '20161202'
                                  , K                     = 1600.0
                                  , nb_sim                = 10000
                                  , rho                   = 0.95
                                  , simplify_compute      = simplify_compute
                                  , return_flight         = True)
        print v1[0]

        self.assertTrue(True)

    def test_ao_new2( self
                    , simplify_compute = 'take_last_only' ):

        v1 = ao.compute_option_val( option_start_date   = '20180202'
                                  , option_end_date     = '20180202'
                                  , outbound_date_start = '2018-03-01'
                                  , outbound_date_end   = '2018-03-02'
                                  , K                   = 150.0
                                  , nb_sim              = 30000
                                  , rho                 = 0.95
                                  , simplify_compute    = simplify_compute
                                  , return_flight       = False )
        print v1[0]

        self.assertTrue(True)

    def test_ao_new3( self
                    , simplify_compute = 'take_last_only'
                    , nb_sim           = 10000
                    , cuda_ind         = False):

        v1 = ao.compute_option_val( option_start_date     = '20180301'
                                  , option_end_date       = '20180301'
                                  , option_ret_start_date = '20180302'
                                  , option_ret_end_date   = '20180302'
                                  , K                     = 100.0
                                  , nb_sim                = nb_sim
                                  , rho                   = 0.95
                                  , simplify_compute      = simplify_compute
                                  , underlyer             = 'n'
                                  , return_flight         = False
                                  , cuda_ind              = cuda_ind)
        print v1[0]

        self.assertTrue(True)

    def test_ao_rho(self):
        """
        tests the effect of correlation

        """
        rho_l               = [0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.2, -0.2, -0.3, -0.9]
        outbound_date_start = '2018-06-07'
        outbound_date_end   = '2018-06-09'
        inbound_date_start  = '2018-06-20'
        inbound_date_end    = '2018-06-21'
        K                   = 1000.
        impact_rho = {}
        for rho in rho_l:
            k1 = ao.compute_option_val(outbound_date_start = outbound_date_start,
                                       outbound_date_end   = outbound_date_end,
                                       inbound_date_start  = inbound_date_start,
                                       inbound_date_end    = inbound_date_end,
                                       K                   = K,
                                       return_flight       = True,
                                       rho                 = rho)
            impact_rho[rho] = k1[1]
            print "RHO ", rho, ":", k1[1]
            print "Impact:", impact_rho

        self.assertTrue(True)

    def test_get_flight_data(self):
        """
        Tests whether the get_flight_data function even executes
        """

        # outbound date
        date_out_dash = ds.convert_date_datedash(ao.date_today() + dt.timedelta(days=30))
        # inbound date
        date_in_dash  = ds.convert_date_datedash(ao.date_today() + dt.timedelta(days=45))

        res = ao.get_flight_data( outbound_date_start = date_out_dash
                                , outbound_date_end   = date_out_dash
                                , inbound_date_start  = date_in_dash
                                , inbound_date_end    = date_in_dash )
        print res

        self.assertTrue(True)
