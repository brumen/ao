# test cases for the monte-carlo module
import config
import numpy as np
import unittest
import mc
from vols.vols import corr_hyp_sec_mat
import air_option as ao

if config.CUDA_PRESENT:
    import pycuda.gpuarray as gpa


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
