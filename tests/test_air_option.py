# testing framework for air options

import numpy as np
import datetime
from unittest import TestCase

import ao.ds          as ds
import ao.air_option  as airo
import ao.ao_estimate as airoe


from ao.air_option import AirOptionFlights, AirOptionMock, AirOptionSkyScanner, AirOptionsFlightsExplicitSky


class TestAirOptionFlights(TestCase):

    def test_basic(self):
        """ Tests if AirOptionFlights even runs, and tests some characteristics of option value
        """

        flights = [(100., datetime.date(2019, 7, 15), 'UA70'), (200., datetime.date(2019, 7, 20), 'UA71')]

        airof = AirOptionFlights( datetime.date(2019, 6, 28)
                              , flights
                              , K=200. )

        airof_pv01 = airof.PV01()  # TODO: DO A TEST w/ this
        self.assertGreater(airof.PV(), 0.)  # air option value > 0 test, silly test

        flights_ret = [(150., datetime.date(2019, 7, 22), 'UA72'), (250., datetime.date(2019, 7, 25), 'UA73')]

        airof2 = AirOptionFlights(datetime.date(2019, 6, 28), (flights, flights_ret), K=200.)

        airof2_pv01 = airof2.PV01()
        self.assertGreater(airof2.PV(), 0.)

        # option times
        # option_range_1 = airof.option_range([datetime.date(2019, 6, 30), datetime.date(2019, 7, 2)])

        # call w/ option_times
        option_range_2 = airof.PV(option_maturities=(datetime.date(2019, 6, 30), datetime.date(2019, 7, 2)))
        option_range_3 = airof2.PV(option_maturities= (datetime.date(2019, 6, 30), datetime.date(2019, 7, 2)))

        # TODO: MORE HERE!!!
        self.assertTrue(True)

    def test_extreme(self):
        """ Tests if AirOptionFlights runs with multiple flights.
        """

        nb_dep_flights = 50
        dep_flights = list(zip( list(range(100, 100 + nb_dep_flights))
                     , [datetime.date(2019, 7, 1) + datetime.timedelta(days=day_diff) for day_diff in range(nb_dep_flights)]
                     , ['UA' + str(flight_nb) for flight_nb in range(nb_dep_flights)] ))

        ret_flights = list(zip( list(range(100, 100 + nb_dep_flights))
                     , [datetime.date(2019, 10, 1) + datetime.timedelta(days=day_diff) for day_diff in range(nb_dep_flights)]
                     , ['UA' + str(flight_nb) for flight_nb in range(nb_dep_flights)] ))

        airof = AirOptionFlights( datetime.date(2019, 6, 1)
                              , (dep_flights, ret_flights)
                              , K=300. )

        res1 = airof.PV(option_maturities=(datetime.date(2019, 6, 10), datetime.date(2019, 6, 15), datetime.date(2019, 6, 20)))  # air option value

        self.assertTrue(True)


class TestAirOptionSkyscanner(TestCase):
    """ Tests the skyscanner version of the air option.

    """

    def test_1(self):
        ao_ss = AirOptionSkyScanner( datetime.date(2017, 2, 1)
                                   , origin = 'SFO'
                                   , dest   = 'EWR'
                                   , outbound_date_start = datetime.date(2017, 4, 26)
                                   , outbound_date_end   = datetime.date(2016, 4, 26)
                                   , K                   = 100.
                                   , carrier='UA' )

        self.assertGreater(ao_ss.PV(), 0.)


class TestAirOptionsFlightsExplicitSky(TestCase):
    def test_1(self):
        ao_ss = AirOptionsFlightsExplicitSky( datetime.date(2016, 9, 25)
                                   , origin = 'SFO'
                                   , dest   = 'EWR'
                                   , outbound_date_start = datetime.date(2016, 10, 1)
                                   , outbound_date_end   = datetime.date(2016, 10, 4)
                                   , K                   = 100.
                                   , carrier='UA' )

        pv = ao_ss.PV()

        self.assertGreater(pv, 0.)


class TestAirOptionMock(TestCase):

    def test_1(self):
        """ Checking if the mock air option computation works.
        """

        airom = AirOptionMock( datetime.date(2019, 7, 2)
                           , origin = 'SFO'
                           , dest = 'EWR'
                           , K = 1600.)

        self.assertGreaterEqual(airom.PV(), 0.)


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

    def test_airo_2( self
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

        res = airo.obtain_flights( 'EWR'
                               , 'SFO'
                               , 'UA'
                               , self.io_dr
                               , None)

        self.assertTrue(True)

    def test_airo_26(self, nb_sim=10000):
        """
        tests the air option
        """
        tickets = np.linspace(450., 400., 50)
        s_v = 100. * np.ones(len(tickets))
        T_l = np.linspace(1./365., 180./365., 180)
        T_mat = T_l + 0.01  # some increase over sim_times

        res = AirOption.air_option( tickets
                                  , s_v
                                  , s_v
                                  , T_l
                                  , T_mat
                                  , 100.
                                  , nb_sim = nb_sim)

        self.assertTrue(True)

    def test_airo_27( self
                  , nb_sim=10000):
        """
        tests the air option

        """

        tickets = np.linspace(450., 400., 3)
        s_v = 100. * np.ones(len(tickets))
        T_l = np.linspace(1./365., 180./365., 2)
        d_v = 0.2 * np.ones(len(tickets))

        res = airo.air_option( tickets
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

        print(airoe.all_vols_by_airline(airline, use_cache=True))
        self.assertTrue(True)

    def test_simple_return(self):
        """
        Tests the compute_option_val function on return flight

        """

        v1 = airo.compute_option_val( origin_place          = 'SFO'
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

        v1 = airo.compute_option_val( origin_place          = 'SFO'
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

        v1 = airo.compute_option_val( origin_place          = 'SFO'
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

    def test_airo_new2( self
                    , simplify_compute = 'take_last_only'
                    , cuda_ind         = False ):

        v1 = airo.compute_option_val( option_start_date   = self.optionOutDateStart
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

    def test_airo_rho(self):
        """
        tests the effect of correlation

        """
        rho_l      = [0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.2, -0.2, -0.3, -0.9]
        K          = 1000.
        impact_rho = {}
        for rho in rho_l:
            k1 = airo.compute_option_val( outbound_date_start = self.outDate
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


if __name__ == "__main__":
    unittest.main()
