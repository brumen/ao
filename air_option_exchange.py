# exchange version of the air option

import datetime
import numpy as np
from typing import Union, Tuple

import mc

from air_option import AirOptionFlights
from vols.vols  import corr_hyp_sec_mat


class AirOptionFlightsExchange(AirOptionFlights):
    """ Handles the air option for a particular set of flights.

        Important: Parameter K is replaced by an actual flight and
        used in simulation, computation.
    """

    @property
    def __strike_fwd(self) -> float:
        """ Forward value associated w/ the strike flight.
        """

        return 1.

    @property
    def __strike_s(self) -> float:
        return 1.

    @property
    def __strike_d(self) -> float:
        return 2.

    @property
    def __strike_F_mat(self) -> float:
        return 3.

    def _air_option_sims(self
                         , sim_times : Union[np.array, Tuple[np.array, np.array]]
                         , nb_sim    = 1000
                         , rho       = 0.9
                         , cuda_ind  = False
                         , underlyer ='n'
                         , keep_all_sims = False):
        """ Parameters the same as in the base class.
        """

        return_flight_ind = isinstance(self._F_v, tuple)

        F_v = self._F_v + [self.__strike_fwd]

        if return_flight_ind:
            # correlation matrix for departing, returning flights

            rho_m = ( corr_hyp_sec_mat(rho, range(len(F_v[0])))
                    , corr_hyp_sec_mat(rho, range(len(F_v[1]))) )

        else:  # only outgoing flight
            rho_m = corr_hyp_sec_mat(rho, range(len(F_v)))

        # which monte-carlo method to use.
        mc_used = mc.mc_mult_steps if not return_flight_ind else mc.mc_mult_steps_ret

        s_v     = self._s_v + [self.__strike_s]
        d_v     = self._d_v + [self.__strike_d]
        F_mat_v = self._F_mat_v + [self.__strike_F_mat]

        return mc_used( F_v
                      , s_v
                      , d_v
                      , sim_times
                      , rho_m
                      , F_mat_v
                      , nb_sim   = nb_sim
                      , model    = underlyer
                      , cuda_ind = cuda_ind
                      , keep_all_sims= keep_all_sims)

    def air_option(self
                   , sim_times : Union[np.array, Tuple[np.array, np.array]]
                   , K         : float
                   , nb_sim    = 1000
                   , rho       = 0.9
                   , cuda_ind  = False
                   , underlyer ='n'
                   , keep_all_sims = False):

        """ Parameters the same as in the base.
        """

        F_all = self._air_option_sims(sim_times
                                      , nb_sim    = nb_sim
                                      , rho       = rho
                                      , cuda_ind  = cuda_ind
                                      , underlyer = underlyer
                                      , keep_all_sims = keep_all_sims)

        # Importnt: These two lines differ from the previous one.
        F_max = F_all[:, :-1]  # all other forward prices
        K     = F_all[:, -1]  # last column is simulated strike

        # final option payoff
        if not keep_all_sims:
            _, F_sim_realized = list(F_max)[0]
            return np.mean(np.maximum (np.amax(F_sim_realized, axis=0) - K, 0.))

        # keep all simulation case
        return {sim_time: np.mean(np.maximum (np.amax(F_max_at_time, axis=0) - K, 0.))
                for sim_time, F_max_at_time in F_max}
