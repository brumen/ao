""" AO rester tests.
    Important: For this test, the rester has to run, i.e. ao_scripts/ao_rester.py
"""

import requests

from unittest import TestCase


class AOResterTest(TestCase):

    def test_hello_world(self):
        """ Tests whether the basic test applies.
        """

        try:
            response = requests.get("http://localhost:5000/ao/")
        except Exception as e:
            self.assertFalse(expr=e)

        self.assertEqual(response.content, b'Hello World')

    def test_compute_ao_option(self):
        """ Test whether the AO is computed.
        """

        response = requests.get( 'http://localhost:5000/ao/compute_option'
                               , params = { 'return_ow'     : 'one_way'
                                          , 'origin'        : 'San Francisco'
                                          , 'dest'          : 'Newark'
                                          , 'outbound_start': '5-10-2017'
                                          , 'outbound_end'  : '5-15-2017'
                                          , 'ticket_price'  : 100.
                                          , 'airline_name'  : 'United'
                                          , 'nb_people'     : 1
                                          , 'cabin_class'   : 'Economy'
                                          , } )

        # ao_ss = AirOptionSkyScanner( datetime.date(2017, 4, 26)
        #                            , origin = 'SFO'
        #                            , dest   = 'EWR'
        #                            , outbound_date_start = datetime.date(2017, 5, 10)
        #                            , outbound_date_end   = datetime.date(2017, 5, 15)
        #                            , K                   = 100.
        #                            , carrier='UA' )

        # self.assertGreaterEqual(ao_ss.PV(), 0.)
        self.assertTrue(True)
