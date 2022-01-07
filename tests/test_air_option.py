""" testing framework for air options
"""

import datetime
from unittest import TestCase

from ao.air_option        import AirOptionFlights
from ao.air_option_derive import AirOptionMock, AirOptionSkyScanner, AirOptionsFlightsExplicitSky


class TestAirOptionFlights(TestCase):

    def test_basic(self):
        """ Tests if AirOptionFlights even runs, and tests some characteristics of option value
        """

        flights = [(100., datetime.date(2019, 7, 15), 'UA70'), (200., datetime.date(2019, 7, 20), 'UA71')]

        airof = AirOptionFlights( datetime.date(2019, 6, 28)
                                , flights
                                , K = 200. )

        airof_pv01 = airof.PV01()  # TODO: DO A TEST w/ this
        self.assertGreater(airof.PV(), 0.)  # air option value > 0 test, silly test

        flights_ret = [(150., datetime.date(2019, 7, 22), 'UA72'), (250., datetime.date(2019, 7, 25), 'UA73')]

        airof2 = AirOptionFlights(datetime.date(2019, 6, 28), (flights, flights_ret), K=200.)

        airof2_pv01 = airof2.PV01()
        self.assertGreater(airof2.PV(), 0.)

        # call w/ option_times
        option_range_2 = airof.PV(option_maturities=[datetime.date(2019, 6, 30), datetime.date(2019, 7, 2)])
        option_range_3 = airof2.PV(option_maturities= [datetime.date(2019, 6, 30), datetime.date(2019, 7, 2)])
        option_range_4 = airof.PV(option_maturities=[datetime.date(2019, 8, 1)])

        # option values are increasing in maturity
        self.assertGreaterEqual(option_range_2[datetime.date(2019, 6, 30)], option_range_2[datetime.date(2019, 7, 2)])
        self.assertGreaterEqual(option_range_3[datetime.date(2019, 6, 30)], option_range_2[datetime.date(2019, 7, 2)])

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

        res1 = airof.PV(option_maturities=[datetime.date(2019, 6, 10), datetime.date(2019, 6, 15), datetime.date(2019, 6, 20)])  # air option value

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

        self.assertGreaterEqual(ao_ss.PV(), 0.)

    def test_2(self):
        ao_ss = AirOptionSkyScanner( datetime.date(2017, 4, 26)
                                   , origin = 'SFO'
                                   , dest   = 'EWR'
                                   , outbound_date_start = datetime.date(2017, 5, 10)
                                   , outbound_date_end   = datetime.date(2017, 5, 15)
                                   , K                   = 100.
                                   , carrier='UA' )

        self.assertIsNotNone(ao_ss.flights)  # this should not be empty with the current database.
        self.assertGreaterEqual(ao_ss.PV(), 0.)


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

        self.assertGreaterEqual(pv, 0.)


class TestAirOptionMock(TestCase):

    def test_1(self):
        """ Checking if the mock air option computation works.
        """

        airom = AirOptionMock( datetime.date(2019, 7, 2)
                           , origin = 'SFO'
                           , dest = 'EWR'
                           , K = 1600.)

        self.assertGreaterEqual(airom.PV(), 0.)
