# testing framework for air options

import numpy    as np
import datetime
from unittest import TestCase


import ds
import air_option  as ao
import ao_estimate as aoe

from air_option import AirOptionFlights, AirOptionSkyScanner, AirOptionMock


class TestAirOptionFlights(TestCase):

    def test_basic(self):
        """ Tests if AirOptionFlights even runs.

        """

        flights = [(100., datetime.date(2019, 7, 15), 'UA70'), (200., datetime.date(2019, 7, 20), 'UA71')]

        aof = AirOptionFlights( datetime.date(2019, 6, 28)
                              , flights
                              , K=1600. )

        res1 = aof()  # air option value
        self.assertGreater(res1, 0.)

        flights_ret = [(150., datetime.date(2019, 7, 22), 'UA72'), (250., datetime.date(2019, 7, 25), 'UA73')]

        aof2 = AirOptionFlights(datetime.date(2019, 6, 28), (flights, flights_ret), K=200.)

        res2 = aof2()

        self.assertGreater(res2, 0.)

        # option times
        # option_range_1 = aof.option_range([datetime.date(2019, 6, 30), datetime.date(2019, 7, 2)])

        # call w/ option_times
        option_range_2 = aof(option_maturities=[datetime.date(2019, 6, 30), datetime.date(2019, 7, 2)])
        option_range_3 = aof2(option_maturities= [datetime.date(2019, 6, 30), datetime.date(2019, 7, 2)])

        self.assertTrue(True)

    def test_extreme(self):
        """ Tests if AirOptionFlights runs with multiple flights.
        """

        import time

        nb_dep_flights = 50
        dep_flights = list(zip( list(range(100, 100 + nb_dep_flights))
                     , [datetime.date(2019, 7, 1) + datetime.timedelta(days=day_diff) for day_diff in range(nb_dep_flights)]
                     , ['UA' + str(flight_nb) for flight_nb in range(nb_dep_flights)] ))

        ret_flights = list(zip( list(range(100, 100 + nb_dep_flights))
                     , [datetime.date(2019, 10, 1) + datetime.timedelta(days=day_diff) for day_diff in range(nb_dep_flights)]
                     , ['UA' + str(flight_nb) for flight_nb in range(nb_dep_flights)] ))

        aof = AirOptionFlights( datetime.date(2019, 6, 1)
                              , (dep_flights, ret_flights)
                              , K=300.
                              , nb_sim=100000  )

        t1 = time.time()
        res1 = aof(option_maturities=[datetime.date(2019, 6, 10), datetime.date(2019, 6, 15), datetime.date(2019, 6, 20)])  # air option value
        print(time.time() - t1)

        self.assertTrue(True)

    def test_4(self):
        """ Testing the compute_option_raw function.

        """

        nb_dep, nb_ret = 100, 100
        F_v = (np.linspace(100., 150., nb_dep), np.linspace(100., 150., nb_ret))
        s_v = ( np.linspace(0.2, 1.4, nb_dep)
              , np.linspace(0.2, 1.4, nb_ret) )
        d_v = ( np.linspace(0.2, 1.4, nb_dep)
              , np.linspace(0.2, 1.4, nb_ret) )
        T_v_exp = ( np.linspace(0.9, 1., nb_dep)
                  , np.linspace(1.1, 1.2, nb_ret) )
        T_l = ( np.array([0.55, 0.62, 0.73])
              , np.array([0.55, 0.62, 0.73]) )
        rho = 0.99

        res1 = AirOptionFlights.compute_option_raw( F_v
                                                 , s_v
                                                 , d_v
                                                 , T_l
                                                 , T_v_exp
                                                 , 400.  # K
                                                 , rho )

        res2 = AirOptionFlights.compute_option_raw( F_v
                                                 , s_v
                                                 , d_v
                                                 , T_l
                                                 , T_v_exp
                                                 , 500.  # K
                                                 , rho )

        self.assertGreater(res1, res2)


class TestAirOptionMock(TestCase):

    def test_1(self):
        """ Checking if the mock air option computation works.
        """
        aom = AirOptionMock( datetime.date(2019, 7, 2)
                           , origin = 'SFO'
                           , dest = 'EWR'
                           # when can you change the option
                           , K = 1600.
                           , nb_sim = 10000 )

        res1 = aom()

        self.assertTrue(True)


class TestAirOption(TestCase):

    def setUp(self):
        """
        Setting of the basic variables.

        """

        self.outDate            = datetime.date.today() + datetime.timedelta( days = 30 )
        self.outDatePlusOne     = self.outDate + datetime.timedelta( days = 1 )
        self.optionOutDateStart = datetime.date.today() + datetime.timedelta ( days = 1 )
        self.optionOutDateEnd   = self.outDate - datetime.timedelta ( days = 1 )

        self.retDate            = self.outDate + datetime.timedelta ( days = 7 )
        self.retDatePlusOne     = self.retDate + datetime.timedelta ( days = 1 )
        self.optionRetDateStart = datetime.date.today() + datetime.timedelta ( days = 1 )
        self.optionRetDateEnd   = self.outDate - datetime.timedelta ( days = 1 )

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
        res = AirOption.air_option( tickets
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

    def test_ao_26(self, nb_sim=10000):
        """
        tests the air option
        """
        tickets = np.linspace(450., 400., 50)
        s_v = 100. * np.ones(len(tickets))
        T_l = np.linspace(1./365., 180./365., 180)
        T_mat = T_l + 0.01  # some increase over T_l

        res = AirOption.air_option( tickets
                                  , s_v
                                  , s_v
                                  , T_l
                                  , T_mat
                                  , 100.
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
                           , nb_sim = nb_sim)

        self.assertTrue(True)

    def test_compute_vols( self
                         , airline = 'Alaska Airlines' ):
        """
        airline from the cache database  # TODO: THIS FUNCTION DOES NOT WORK

        """

        print(aoe.all_vols_by_airline(airline, use_cache=True))
        self.assertTrue(True)

    def test_simple_return(self):
        """
        Tests the compute_option_val function on return flight

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
                                  , simplify_compute      = 'take_last_only'
                                  , underlyer             = 'n'
                                  , price_by_range        = False
                                  , return_flight         = True )
        print(v1)
        self. assertTrue(True)

    def test_simple_one_way(self):
        """
        Tests the compute_option_val function on one-way flight

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
                                  , simplify_compute      = 'take_last_only'
                                  , underlyer             = 'n'
                                  , price_by_range        = True
                                  , return_flight         = False )
        print(v1)
        self. assertTrue(True)

    def test_simple_fixed(self):
        """
        Tests the compute_option_val function

        """

        v1 = ao.compute_option_val( origin_place          = 'SFO'
                                  , dest_place            = 'EWR'
                                  , option_start_date     = datetime.date(2018, 2, 25)
                                  , option_end_date       = datetime.date(2018, 5, 15)
                                  , option_ret_start_date = datetime.date(2018, 2, 25)
                                  , option_ret_end_date   = datetime.date(2018, 7, 10)
                                  , outbound_date_start   = datetime.date(2018, 5, 16)
                                  , outbound_date_end     = datetime.date(2018, 5, 17)
                                  , inbound_date_start    = datetime.date(2018, 7, 12)
                                  , inbound_date_end      = datetime.date(2018, 7, 13)
                                  , K                     = 1240.0
                                  , carrier               = 'UA'
                                  , nb_sim                = 10000
                                  , rho                   = 0.95
                                  , adults                = 1
                                  , cuda_ind              = False
                                  , simplify_compute      = 'take_last_only'
                                  , underlyer             = 'n'
                                  , price_by_range        = True
                                  , return_flight         = True )
        print(v1)
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
