# test for ao_params
import unittest
import ao_params as aop


class TestAoParams(unittest.TestCase):

    def test_get_drift_vol_from_db(self):
        res = aop.get_drift_vol_from_db( '2017-03-01'
                                       , 'SFO'
                                       , 'EWR'
                                       , 'UA')
        print res

        self.assertTrue(True)

    def test_get_drift_vol_from_db_precise(self):
        res = aop.get_drift_vol_from_db_precise( ['2018-03-02T06:00:00']
                                               , 'SFO'
                                               , 'EWR'
                                               , 'UA')
        print res

        self.assertTrue(True)
