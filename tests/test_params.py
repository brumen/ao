# test for ao_params
import unittest
import datetime

import ao_params as aop


class TestAoParams(unittest.TestCase):

    def test_get_drift_vol_from_db(self):
        """ Tests if we can get any drift from database.

        """

        res = aop.get_drift_vol_from_db( datetime.date(2017, 3, 1), 'SFO', 'EWR', 'UA')

        self.assertTrue(True)

    def test_get_drift_vol_from_db_precise(self):
        """ Tests for the precise function.

        """

        res = aop.get_drift_vol_from_db_precise( ['2018-03-02T06:00:00']
                                               , 'SFO'
                                               , 'EWR'
                                               , 'UA')

        self.assertTrue(True)
