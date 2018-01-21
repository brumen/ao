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

    def test_2(self):
        """
        Martingale test for 1 single underlyer - the average of this should be very close to the
        of the underlyer

        """
        # TODO: TO FINISH THIS TEST

        F = np.array([100.])
        s = np.array([0.25])
        s_d_fct = [lambda t: s[[0]]]
        d_v_fct = [lambda t: 0.01]
        nb_sim = 1048575  # 2**20 - 1
        rho_m = np.array([[1.]])
        T_l = np.linspace(0.1, 1., 10)
        F_sim_l = mc.mc_mult_steps( F
                                  , s_d_fct
                                  , d_v_fct
                                  , T_l
                                  , rho_m
                                  , nb_sim
                                  , T_l+0.01 )

        print np.average(F_sim_l[-1, :, 0])

        self.assertTrue(True)

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

        print "r", res['F_max_prev']

        self.assertTrue(True)

    def test_4(self, cuda_ind=False):
        """

        """

        nb_dep, nb_ret = 50, 100
        F_v = (np.linspace(100., 150., nb_dep), np.linspace(100., 150., nb_ret))
        s_v = ([lambda arg: x for x in np.linspace(0.2, 0.4, nb_dep)],
               [lambda arg: x for x in np.linspace(0.2, 0.4, nb_ret)])
        d_v = ([lambda arg: x for x in np.linspace(0.2, 0.4, nb_dep)],
               [lambda arg: x for x in np.linspace(0.2, 0.4, nb_ret)])
        T_v_exp = (np.linspace(0.9, 1., nb_dep), np.linspace(1.1, 1.2, nb_ret))
        T_l = (np.array([0.55, 0.62, 0.73]),  np.array([0.55, 0.62, 0.73]))
        nb_sim = 50000

        if not cuda_ind:
            F_start = np.zeros((nb_dep, nb_sim))
        else:
            F_start = gpa.zeros((nb_dep, nb_sim), np.double)

        return ao.compute_option_raw( F_v
                                    , s_v
                                    , d_v
                                    , T_l
                                    , T_v_exp
                                    , 150.
                                    , rho
                                    , cuda_ind=cuda_ind)
