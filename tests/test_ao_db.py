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
        tests whether the mc_mult_steps_cpu function works

        """

        F_v = np.zeros(nb_tickets) + 100.
        s_v = np.zeros(nb_tickets) + 0.2
        d_v = s_v
        rho_m = corr_hyp_sec_mat(0.95, range(nb_tickets))
        v1 = mc.mc_mult_steps( F_v
                             , s_v
                             , d_v
                             , sim_times
                             , rho_m
                             , nb_sim
                             , sim_times + 0.01
                             , ao_f = ao.ao_f_arb)
        print "V1:", v1
        self.assertTrue(True)
