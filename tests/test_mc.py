# test cases for the monte-carlo module
import config
import numpy as np
import unittest
import mc
from vols.vols import corr_hyp_sec_mat
import air_option as ao

if config.CUDA_PRESENT:
    import pycuda.gpuarray as gpa


class TestMC(unittest.TestCase):
    """
    Testing of the Monte-Carlo part of AirOption

    """

    def setUp(self):

        self.nb_tickets = 5
        # mostly used for one-way flights
        self.F_v = np.linspace(100., 200., self.nb_tickets)
        self.s_v = np.linspace(0.2 , 0.6 , self.nb_tickets)
        self.d_v = self.s_v
        self.rho_m = corr_hyp_sec_mat(0.95, range(self.nb_tickets))
        self.F_exp = np.linspace(1, 2, self.nb_tickets)  # expiry times

    def test_0(self):
        """
        Tests whether the mc_mult_steps_cpu function works

        """

        nb_sim = 1000
        sim_times = np.array([0.2, 0.3, 0.4])
        v1 = mc.mc_mult_steps( self.F_v
                             , self.s_v
                             , self.d_v
                             , sim_times
                             , self.rho_m
                             , nb_sim
                             , np.ones(self.nb_tickets) )  # this last par. is maturity of forwards
        print ("V1:", v1)
        self.assertTrue(True)

    def test_2(self):
        """
        Martingale test for underlyers.

        """

        T_l     = np.linspace(0.1, 1., 10)  # forward expiry times
        nb_sim  = 50000
        F_sim_l = mc.mc_mult_steps( self.F_v
                                  , self.s_v
                                  , np.zeros(self.nb_tickets)
                                  , T_l
                                  , self.rho_m
                                  , nb_sim
                                  , self.F_exp )

        self.assertTrue(np.abs(np.average(F_sim_l[-1, :]) - self.F_v[-1]) < 5)

    def test_3(self):
        """
        tests the mc_mult_steps_cpu_ret function for return flights

        """

        F_v = ( np.array([100., 105., 106.])
              , np.array([200., 205.]) )
        s_v = ( np.array([0.2, 0.2, 0.2])
              , np.array([0.2, 0.3]) )

        T_l = ( [0.1, 0.2, 0.3, 0.4, 0.5]
              , [0.15, 0.25, 0.35, 0.45, 0.55] )
        rho_m = ( corr_hyp_sec_mat(0.95, range(3))
                , corr_hyp_sec_mat(0.95, range(2)) )
        nb_sim = 1000
        ao_p = {'model': 'max',
                'F_max_prev': np.zeros((2, nb_sim)),
                'K': 200.,
                'penalty': 100.,
                'P_arg_max': 0.}

        res = mc.mc_mult_steps_ret( F_v
                                  , s_v
                                  , s_v  # d_v = s_v TODO: FIX
                                  , T_l
                                  , rho_m
                                  , nb_sim
                                  , ao_f     = ao.ao_f_arb
                                  , ao_p     = ao_p
                                  , cva_vals = None
                                  , model    = 'n')

        self.assertTrue(True)

    def test_4( self
              , cuda_ind = False):
        """
        Testing the compute_option_raw function.

        """

        nb_dep, nb_ret = 50, 100
        F_v = (np.linspace(100., 150., nb_dep), np.linspace(100., 150., nb_ret))
        s_v = ( np.linspace(0.2, 0.4, nb_dep)
              , np.linspace(0.2, 0.4, nb_ret) )
        d_v = ( np.linspace(0.2, 0.4, nb_dep)
              , np.linspace(0.2, 0.4, nb_ret) )
        T_v_exp = ( np.linspace(0.9, 1., nb_dep)
                  , np.linspace(1.1, 1.2, nb_ret) )
        T_l = ( np.array([0.55, 0.62, 0.73])
              , np.array([0.55, 0.62, 0.73]) )

        res = ao.compute_option_raw( F_v
                                   , s_v
                                   , d_v
                                   , T_l
                                   , T_v_exp
                                   , 150.  # K
                                   , rho
                                   , cuda_ind = cuda_ind)

        self.assertTrue(True)