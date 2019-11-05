# exchange version of the air option

import datetime
import numpy as np

from typing import Union, Tuple

import mc

# from air_flights import get_flight_data
from air_option  import AirOptionFlights
from vols.vols   import corr_hyp_sec_mat
from ao_params   import get_drift_vol_from_db

from air_option import AirOptionSkyScanner


class AirOptionFlightsExchange(AirOptionFlights):
    """ Handles the air option for a particular set of flights.

        Important: Parameter K is replaced by an actual flight and
        used in simulation, computation.
    """

    # TODO: FINISH THIS 4 PROPERTIES HERE!!!

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


class AirOptionSkyScannerExchange(AirOptionFlightsExchange, AirOptionSkyScanner):
    """ Class for handling the air options from SkyScanner inputs for exchange option.
    """

    def __init__( self
                , mkt_date
                , origin    = 'SFO'
                , dest      = 'EWR'
                # next 4 - when do the (changed) flights occur
                , outbound_date_start = None  # departing flights info.
                , outbound_date_end   = None
                , inbound_date_start  = None  # returning flights info
                , inbound_date_end    = None
                , K                   = ('UA96', datetime.date(2018, 1, 1))
                , carrier             = 'UA'
                , nb_sim              = 10000
                , rho                 = 0.95
                , adults              = 1
                , cabinclass          = 'Economy'
                , cuda_ind            = False
                , simplify_compute    = 'take_last_only'
                , underlyer           = 'n'
                , return_flight       = False
                , recompute_ind       = False
                , correct_drift       = True ):
        """ Computes the air option from the data provided.

        :param origin: IATA code of the origin airport ('SFO')
        :param dest: IATA code of the destination airport ('EWR')
        :param outbound_date_start: start date for outbound flights to change to
        :type outbound_date_start: datetime.date
        :param outbound_date_end: end date for outbound flights to change to
        :type outbound_date_end: datetime.date
        :param inbound_date_start: start date for inbound flights to change to
        :type inbound_date_start: datetime.date
        :param inbound_date_end: end date for inbound flights to change to
        :type inbound_date_end: datetime.date
        :param K: option strike
        :type K: double
        :param carrier: IATA code of the carrier
        :type carrier: str
        :param nb_sim: number of simulations
        :type nb_sim: int
        :param rho: correlation between flights parameter
        :type rho: double
        :param adults: nb. of people on this ticket
        :type adults: int
        :param cabinclass: class of flight ticket
        :type cabinclass: str
        :param cuda_ind: whether to use cuda for computation
        :type cuda_ind: bool
        :param simplify_compute: simplifies the computation in that it only simulates the last simulation date
        :type simplify_compute: str, options are: "take_last_only", "all_sim_dates"
        """

        self.mkt_date = mkt_date  # TODO: THIS IS NOT PARTICULARLY CLEAN, AS THIS IS RE-ASSIGNED in the subclass
        self.__origin = origin
        self.__dest   = dest
        self.__outbound_date_start = outbound_date_start
        self.__outbound_date_end   = outbound_date_end
        self.__inbound_date_start  = inbound_date_start
        self.__inbound_date_end    = inbound_date_end
        self.__carrier             = carrier
        self.__cabinclass          = cabinclass
        self.__adults              = adults
        self.__return_flight       = return_flight
        self.__correct_drift       = correct_drift
        self.__recompute_ind       = recompute_ind

        super().__init__( mkt_date              = self.mkt_date
                        , flights               = list(self.get_flights())
                        , cuda_ind              = cuda_ind
                        , rho                   = rho
                        , nb_sim                = nb_sim
                        , K                     = K
                        , simplify_compute      = simplify_compute
                        , underlyer             = underlyer )

        # cache about strike flight
        self.__strike_flight_cache = None

    @property
    def __info_strike_flight(self):
        if self.__strike_flight_cache:
            return self.__strike_flight_cache

        dep_date, flight_id = self.K
        orig, dest, carrier = self._get_origin_dest_carrier_from_flight(flight_id)

        self.__strike_flight_cache =  get_drift_vol_from_db( dep_date
                                    , orig
                                    , dest
                                    , carrier
                                    , default_drift_vol = (500., 501.)
                                    , fwd_value         = None
                                    , db_host           = 'localhost' )

        return self.__strike_flight_cache

    @property
    def __strike_fwd(self) -> float:
        """ Forward value associated w/ the strike flight.
        """

        return self.__info_strike_flight[0]  # TODO: THIS IS COMPLETELY WRONG

    @property
    def __strike_s(self) -> float:
        """ Volatility associated w/ the strike flight.
        """

        return self.__info_strike_flight[1]  # TODO: COMPLETELY WRONG

    @property
    def __strike_d(self) -> float:
        """ Drift associated w/ the strike flight.
        """

        return self.__info_strike_flight[2]  # TODO: COMPLETELY WRONG

    @property
    def __strike_F_mat(self) -> float:
        """Flight maturity associated w/ the strike flight.
        """

        return self.__info_strike_flight[3]  # TODO: COMPLETELY WRONG


class AirOptionMockExchange(AirOptionSkyScannerExchange):
    """ Air options computation for some mock flight data. The data are generated at random.
        IMPORTANT: JUST USED FOR TESTING.
    """

    def get_flights(self):
        """ Generates mock flight data. Is a generator.
        """

        nb_flights = 15

        for flight_nb in range(1, nb_flights):
            yield ( np.random.random() * 100 + 100
                  , self.mkt_date + datetime.timedelta(days=flight_nb)
                  , 'UA' + str(flight_nb) )
