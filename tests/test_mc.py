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

    def test_0(self, nb_sim=100):
        F_d = np.zeros(5) + 100.
        s_d = np.zeros(5) + 0.2
        rho_m = corr_hyp_sec_mat(0.99, range(5)).astype(np.float32)
        v1 = mc.mc_mult_steps(F_d, s_d, [0.2, 0.3], rho_m, nb_sim=nb_sim)
        return v1

    def test_2(self):
        F = np.array([100.])
        T_end = 2.
        nb_sim = 1048575 # 2**20 - 1
        T = np.array ([T_end - 9./12., T_end - 6./12., T_end - 5./12.])
        s = np.array([0.25])
        rho_m = np.array([[1.]])
        T_l = np.linspace (0.1,1., 10)
        F_sim_l = mc.mc_mult_steps(F, s, T_l, rho_m, nb_sim)

        print np.average (F_sim_l[-1,:,0])

        self.assertTrue(True)

    def test_3(self):
        F_v = (np.array([100., 105., 106.]), np.array([200., 205.]))
        s_v = (np.array([0.2, 0.2, 0.2]), np.array([0.2, 0.3]))
        T_l = ([0.1, 0.2, 0.3, 0.4, 0.5], [0.15, 0.25, 0.35, 0.45, 0.55])
        rho_m = (corr_hyp_sec_mat(0.95, range(3)), corr_hyp_sec_mat(0.95, range(2)))
        nb_sim = 1000
        ao_p = {'model': 'max',
                'F_max_prev': np.zeros((2, nb_sim)),
                'K': 200.,
                'penalty': 100.,
                'P_arg_max': 0.}

        res = mc.mc_mult_steps_cpu_ret(F_v, s_v, T_l, rho_m, nb_sim,
                                       ao_f=ao.ao_f_arb,
                                       ao_p=ao_p,
                                       d_v=None, cva_vals=None, model='ln')
        print "r", res['F_max_prev']
        self.assertTrue(True)

    def test_4(self, cuda_ind=False):
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

        return ao.compute_option_raw(F_v, s_v, T_l, T_v_exp, 150., 100., 0.2, 0.95, d_v=d_v,
                                     cuda_ind=cuda_ind)
