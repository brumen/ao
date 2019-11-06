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

    # TODO: THESE 4 PROPERTIES HERE ARE UNIFINISHED
    @property
    def __strike_fwd(self) -> Union[float, Tuple[float, float]]:
        """ Forward value associated w/ the strike flight.
        """

        if not self.return_flight:
            return 1.

        # return flights
        return 1., 1.

    @property
    def __strike_s(self) -> Union[float, Tuple[float, float]]:

        if not self.return_flight:
            return 1.

        # return flights
        return 1., 1.

    @property
    def __strike_d(self) -> Union[float, Tuple[float, float]]:

        if not self.return_flight:
            return 2.

        # return flights
        return 2., 2.

    @property
    def __strike_F_mat(self) -> Union[float, Tuple[float, float]]:

        if not self.return_flight:
            return 3.

        # return flights
        return 3., 3.

    def _air_option_sims(self
                         , sim_times : Union[np.array, Tuple[np.array, np.array]]
                         , nb_sim    = 1000
                         , rho       = 0.9
                         , cuda_ind  = False
                         , underlyer ='n'
                         , keep_all_sims = False):
        """ Parameters the same as in the base class.
        """

        # TODO: Remove this part later.
        # return_flight_ind = isinstance(self._F_v, tuple)

        if not self.return_flight:
            F_v = self._F_v + [self.__strike_fwd]

        else:  # return flight, add to each component individually
            F_v_0, F_v_1 = self._F_v
            F_v_0.append(self.__strike_fwd[0])
            F_v_1.append(self.__strike_fwd[1])
            F_v = (F_v_0, F_v_1)

        if self.return_flight:  # return_flight_ind:
            # correlation matrix for departing, returning flights

            # TODO: CHECK WHY this is highlighted.
            rho_m = ( corr_hyp_sec_mat(rho, range(len(F_v[0])))
                    , corr_hyp_sec_mat(rho, range(len(F_v[1]))) )

        else:  # only outgoing flight
            rho_m = corr_hyp_sec_mat(rho, range(len(F_v)))

        # which monte-carlo method to use.
        mc_used = mc.mc_mult_steps if not self.return_flight else mc.mc_mult_steps_ret

        s_v     = self._s_v     + [self.__strike_s]
        d_v     = self._d_v     + [self.__strike_d]
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

    def air_option( self
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

        # final option payoff
        if not keep_all_sims:
            _, F_max = list(F_all)[0]  # there is only one  observation

            return np.mean(np.maximum (np.amax(F_max[:, :-1], axis=0) - F_max[:, -1].reshape((nb_sim, 1)), 0.))

        # keep all simulation case
        return {sim_time: np.mean(np.maximum (np.amax(F_max_at_time[:, :-1], axis=0) - F_max_at_time[:, -1].reshape((nb_sim, 1)), 0.))
                for sim_time, F_max_at_time in F_all}


class AirOptionSkyScannerExchange(AirOptionFlightsExchange, AirOptionSkyScanner):
    """ Class for handling the air options from SkyScanner inputs for exchange option.
    """

    def __init__( self
                , mkt_date  : datetime.date
                , origin    = 'SFO'
                , dest      = 'EWR'
                # next 4 - when do the (changed) flights occur
                , outbound_date_start = None  # departing flights info.
                , outbound_date_end   = None
                , inbound_date_start  = None  # returning flights info
                , inbound_date_end    = None
                , K                   = ('UA96', datetime.date(2018, 1, 1))
                , carrier             = 'UA'
                , rho                 = 0.95
                , adults              = 1
                , cabinclass          = 'Economy'
                , simplify_compute    = 'take_last_only'
                , underlyer           = 'n'
                , return_flight       = False
                , recompute_ind       = False
                , correct_drift       = True
                , default_drift_vol   = (500., 501.) ):
        """ Computes the air option from the data provided.

        :param mkt_date: market date
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
        :param rho: correlation between flights processes
        :param adults: nb. of people on this ticket
        :param cabinclass: class of flight ticket
        :param cuda_ind: whether to use cuda for computation
        :param simplify_compute: simplifies the computation in that it only simulates the last simulation date
        :type simplify_compute: str, options are: "take_last_only", "all_sim_dates"
        :param default_drift_vol: default drift & vol.
        """

        # market date is assigned in the superclass
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
        self.__default_drift_vol   = default_drift_vol

        super().__init__( mkt_date         = mkt_date
                        , flights          = list(self.get_flights())
                        , K                = K
                        , rho              = rho
                        , simplify_compute = simplify_compute
                        , underlyer        = underlyer )

        # cache about strike flight
        self.__strike_flight_cache = None

    @property
    def __info_strike_flight(self):
        if self.__strike_flight_cache:
            return self.__strike_flight_cache

        dep_date, flight_id = self.K
        orig, dest, carrier = self._get_origin_dest_carrier_from_flight(flight_id)

        self.__strike_flight_cache = get_drift_vol_from_db( dep_date
                                                          , orig
                                                          , dest
                                                          , carrier
                                                          , default_drift_vol = self.__default_drift_vol
                                                          , db_host           = self.db_host )

        return self.__strike_flight_cache

    @property
    def __strike_fwd(self) -> Union[float, Tuple[float, float]]:
        """ Forward value associated w/ the strike flight.
        """

        if not self.return_flight:
            return self.__info_strike_flight[0]  # TODO: CHECK THIS

        info_dep_flight, info_ret_flight = self.__info_strike_flight  # TODO: CHECK THIS
        return info_dep_flight[0], info_ret_flight[0]

    @property
    def __strike_s(self) -> Union[float, Tuple[float, float]]:
        """ Volatility associated w/ the strike flight.
        """

        if not self.return_flight:
            return self.__info_strike_flight[1]  # TODO: COMPLETELY WRONG

        info_dep_flight, info_ret_flight = self.__info_strike_flight  # TODO: CHECK THIS
        return info_dep_flight[1], info_ret_flight[1]

    @property
    def __strike_d(self) -> Union[float, Tuple[float, float]]:
        """ Drift associated w/ the strike flight.
        """

        if not self.return_flight:
            return self.__info_strike_flight[2]  # TODO: COMPLETELY WRONG

        info_dep_flight, info_ret_flight = self.__info_strike_flight  # TODO: CHECK THIS
        return info_dep_flight[2], info_ret_flight[2]

    @property
    def __strike_F_mat(self) -> Union[float, Tuple[float, float]]:
        """Flight maturity associated w/ the strike flight.
        """

        if not self.return_flight:
            return self.__info_strike_flight[3]  # TODO: COMPLETELY WRONG

        info_dep_flight, info_ret_flight = self.__info_strike_flight  # TODO: CHECK THIS
        return info_dep_flight[3], info_ret_flight[3]


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
