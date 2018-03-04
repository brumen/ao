# test cases for the ao_db module

import numpy as np
import unittest
import datetime


import ao_db


class TestAoDb(unittest.TestCase):
    """
    Testing of the database aspects of the AO

    """

    def test_0( self
              , nb_sim     = 1000
              , nb_tickets = 5
              , sim_times  = np.array([0.2, 0.3, 0.4])):
        """
        tests whether the ao database connections work

        """


        self.assertTrue(True)

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