""" Air option computation file
"""

import datetime
import numpy as np
import logging
import functools

from typing      import List, Tuple, Union, Dict, Optional

from ao.mc          import integrate_vol_drift, ln_step, normal_step, vol_drift_vec
from ao.ds          import construct_date_range
from ao.vols.vols   import corr_hyp_sec_mat
from ao.ao_codes    import MIN_PRICE, reserves, tax_rate, ref_base_F
from ao.delta_dict  import DeltaDict
from ao.flight      import Flight, FlightLive, create_session

logging.basicConfig(filename='/tmp/air_option.log')
logger = logging.getLogger(__name__)
logger.setLevel('INFO')


class AOTradeException(Exception):
    pass


FLIGHT_TYPE = Tuple[float, datetime.date, str]


class AirOptionFlights:
    """ Handles the air option for a particular set of flights.
    """

    def __init__( self
                , mkt_date : datetime.date
                , flights  : Union[ None, List[FLIGHT_TYPE]
                                  , Tuple[List[FLIGHT_TYPE], List[FLIGHT_TYPE]]]
                , K                : float = 1600.
                , rho              : float = 0.95
                , simplify_compute : str   = 'take_last_only'
                , underlyer        : str   = 'n' ):
        """ Computes the air option for the flights.

        :param mkt_date: market date
        :param flights: flights to compute the air option over.
                        pairs of (flight forward value, flight_date(forward maturity date), flight_nb )
        :param K: option strike
        :param rho: correlation between flights parameter
        :param simplify_compute: simplifies the computation in that it only simulates the last simulation date,
                                 options are: "take_last_only", "all_sim_dates"
        """

        self.mkt_date  = mkt_date
        self._flights = flights
        self.return_flight = True if isinstance(flights, tuple) else False  # return flight indicator

        self.K = K  # strike price or strike "flight"
        self.__rho = rho
        self.__underlyer = underlyer
        self.__simplify_compute = simplify_compute

        # caching variables
        self.__recompute_option_value = True  # indicator whether the option should be recomputed
        self.__option_value = None

        # other cached vals.
        self.__outbound_flights = None
        self.__inbound_flights  = None

    @classmethod
    def from_flights( cls
                    , mkt_date         : datetime.date
                    , ao_flights       : List[Flight]
                    , strike           : float
                    , rho              : float = 0.95
                    , simplify_compute : str   = 'take_last_only'
                    , underlyer        : str   = 'n'):
        """ Computes the air option from the database.

        :param mkt_date: market date
        :param ao_flights: flights for which air_option is computed.
        :param strike: strike for the air option.
        :param rho: correlation between flights parameter
        :param simplify_compute: simplifies the computation in that it only simulates the last simulation date,
                                 options are: "take_last_only", "all_sim_dates"
        :param underlyer: underlying model to use.
        """

        return cls( mkt_date
                  , [cls.extract_prices(ao_flight) for ao_flight in ao_flights]
                  , K                = strike
                  , rho              = rho
                  , simplify_compute = simplify_compute
                  , underlyer        = underlyer )

    @staticmethod
    def extract_prices(ao_flight : Flight, default_price : float = 200.) -> Tuple[float, datetime.date, str]:
        """ Gets the prices and other data from the flight.

        :param ao_flight: flight information that you want info from.
        :param default_price: default price if the flight is not obtained.
        :returns: triple of (price, flight forward time (term), flight id)
        """

        found_prices = ao_flight.prices  # prices found in the database

        # find the last price, otherwise report a random price
        flight_price = found_prices[-1].price if found_prices else default_price

        return flight_price, ao_flight.dep_date.date(), ao_flight.flight_id_long

    # @classmethod
    # def from_db ( cls
    #             , mkt_date         : datetime.date
    #             , ao_trade_id      : str
    #             , rho              : float = 0.95
    #             , simplify_compute : str   = 'take_last_only'
    #             , underlyer        : str   = 'n'
    #             , session         : Optional[str] = None):
    #     """ Computes the air option from the database.
    #
    #     :param mkt_date: market date
    #     :param ao_trade_id: trade id for a particular AOTrade we want.
    #     :param rho: correlation between flights parameter
    #     :param simplify_compute: simplifies the computation in that it only simulates the last simulation date,
    #                              options are: "take_last_only", "all_sim_dates"
    #     :param underlyer: underlying model to use.
    #     :param session: session used for the fetching of trades database from where the AOTrade is fetched.
    #     """
    #
    #     # database session
    #     session_used = create_session() if session is None else session
    #
    #     ao_trade = session_used.query(AOTrade)\
    #                            .filter_by(position_id=ao_trade_id)\
    #                            .first()  # AOFlight object
    #
    #     if ao_trade is None:
    #         raise AOTradeException(f'Trade number {ao_trade_id} could not be found.')
    #
    #     return cls.from_flights( mkt_date
    #                            , ao_trade.flights
    #                            , strike           = ao_trade.strike
    #                            , rho              = rho
    #                            , simplify_compute = simplify_compute
    #                            , underlyer        = underlyer )

    @property
    def flights(self) -> List:
        return self._flights

    @flights.setter
    def flights(self, new_flights):
        self.__recompute_option_value = True  # recompute everything
        self._flights = new_flights

    @property
    def outbound_flights(self) -> List:
        if self.__outbound_flights:
            return self.__outbound_flights

        self.__outbound_flights = self.flights if not self.return_flight else self.flights[0]
        return self.__outbound_flights

    @property
    def inbound_flights(self) -> List:
        if self.__inbound_flights is not None:
            return self.__inbound_flights

        self.__inbound_flights = self.flights[1] if self.return_flight else []
        return self.__inbound_flights

    @staticmethod
    def construct_sim_times( date_start       : datetime.date
                           , date_end         : datetime.date
                           , mkt_date         : datetime.date
                           , simplify_compute : str   = 'take_last_only'
                           , dcf              : float = 365.25 ) -> List[float]:
        """ Constructs the simulation times used for air options, simulation times are floats as dcf from
            date_today.

        :param date_start: date start of simulated times
        :param date_end: date end of simulated times
        :param mkt_date: market date
        :param dcf: day count factor
        :returns: list of simulated dates
        """

        date_range = construct_date_range(date_start, date_end)

        if simplify_compute == 'all_sim_dates':
            return [(date_sim - mkt_date).days / dcf for date_sim in date_range]

        # elif simplify_compute == 'take_last_only':
        return [(date_range[-1] - mkt_date).days / dcf]

    @functools.lru_cache(maxsize=128)
    def _drift_vol_for_flight(self, flight_nb: Tuple[datetime.date, str]) -> Tuple[float, float]:
        """ Gets drift and vol for flights. It caches it, so that drift and vol do not forget it.

        :param flight_nb: flight number, e.g. 'UA96'
        """

        return 0.3, 0.3

    def _drift_for_flight(self, flight_nb : Tuple[datetime.date, str]) -> float:
        """ Drift for flight, read from the database.

        :param flight_nb: flight number
        :returns: drift for the particular flight
        """

        return self._drift_vol_for_flight(flight_nb)[0]  # Important: this is cached in the function above

    def _vol_for_flight(self, flight_nb : Tuple[datetime.date, str]) -> float:
        """ Drift for flight, this method can be reimplemented.

        :param flight_nb: flight number
        :returns: drift for a particular flight
        """

        return self._drift_vol_for_flight(flight_nb)[1]  # Important: this is cached, so no double getting.

    @staticmethod
    def _get_origin_dest_carrier_from_flight(flight_id : str, session = None ) -> Tuple[str, str, str]:
        """ Gets origin, destination, carrier from flight_id, e.g. 'UA70' is 'SFO', 'EWR', 'UA'
            TODO: THIS IS ONLY PARTIALLY, HACK IMPLEMENTED, IMPLEMENT HERE IF YOU CAN!!!

        :param flight_id: flight id, in the form 'UA70'
        :param flight_session session: session for the database query, if None, one is created on the fly
        :returns: origin airport, destination airport, carrier
        """

        try:
            carrier   = flight_id[:2]  #  iata code is 2 digit.
            flight_nb = flight_id[2:]  # a string, flight nb, like 70

            session_used = create_session() if session is None else session

            ao_flight = session_used.query(FlightLive)\
                                    .filter_by(carrier=carrier, flight_nb=flight_nb)\
                                    .first()  # FlightLive object

            return ao_flight.orig, ao_flight.dest, carrier

        # TODO: THIS IS TO BE IMPROVED HERE.
        except Exception as e:
            logger.warning(f'Could not locate flight {flight_id}: {e}')

        return 'SFO', 'EWR', 'UA'

    @property
    def _F_v(self) -> Union[List[float], Tuple[List[float], List[float]]]:
        """  Flights forward values. Extracts the forward values from the flights.
        For one-way flights it's a single list, for return flights it's a tuple of lists.
        """

        # one-directional flights
        if not self.return_flight:
            return [fwd_value for fwd_value, _, _ in self.flights]

        # return flights
        dep_flights, ret_flights = self.flights
        return [fwd_value for fwd_value, _, _ in dep_flights], [fwd_value for fwd_value, _, _ in ret_flights]

    @property
    def _F_mat_v(self, dcf = 365.25) -> Union[List[float], Tuple[List[float], List[float]]]:
        """ Extracts maturities from the flights.

        :param dcf: day-count factor for transforming it to numerical values.
        """

        # maturity has to be in numeric terms
        if not self.return_flight:
            return [(F_dep_maturity - self.mkt_date).days / dcf for _, F_dep_maturity, _ in self.flights]

        # return flights
        dep_flights, ret_flights = self.flights
        return [(F_dep_maturity - self.mkt_date).days / dcf for _, F_dep_maturity, _ in dep_flights],\
               [(F_dep_maturity - self.mkt_date).days / dcf for _, F_dep_maturity, _ in ret_flights],

    @property
    def _s_v(self) -> Union[List[float], Tuple[List[float], List[float]]]:
        """ Extracts the volatilities for the flights.
        """

        if not self.return_flight:
            return [self._vol_for_flight((F_dep_maturity, flight_nb)) for _, F_dep_maturity, flight_nb in self.flights]

        # return flights
        dep_flights, ret_flights = self.flights
        return [self._vol_for_flight((F_dep_maturity, flight_nb)) for _, F_dep_maturity, flight_nb in dep_flights], \
               [self._vol_for_flight((F_dep_maturity, flight_nb)) for _, F_dep_maturity, flight_nb in ret_flights]

    @property
    def _d_v(self) -> Union[List[float], Tuple[List[float], List[float]]]:
        """ Extracts the list of drifts for the flights.
        """

        if not self.return_flight:
            return [self._drift_for_flight((F_dep_maturity, flight_nb)) for _, F_dep_maturity, flight_nb in self.flights]

        # return flights
        dep_flights, ret_flights = self.flights
        return [self._drift_for_flight((F_dep_maturity, flight_nb)) for _, F_dep_maturity, flight_nb in dep_flights], \
               [self._drift_for_flight((F_dep_maturity, flight_nb)) for _, F_dep_maturity, flight_nb in ret_flights]

    def __dep_ret_sim_times_num( self
                               , option_start_date     = None
                               , option_end_date       = None
                               , option_ret_start_date = None
                               , option_ret_end_date   = None ):
        """ Same as extract_prices_maturities, just that it considers both departing, returning flights
            Returns the following tuple:
               a. In case of departing flights:
                  F_v, F_mat, s_v, d_v, dep_sim_times_num: F_v   - list of departure flight forward values.
                                                           F_mat - list of departure flight maturities
                                                           s_v   - list of dep. flight volatilities
                                                           d_v   - list of dep. flight drifts
                                                           dep_sim_times_num - simulated teimes for departures
               b. In case of return flights:
                  (F_v_dep, F_v_ret), (F_mat_dep, F_mat_ret), (s_v_dep, s_v_ret), (d_v_dep, d_v_ret), (dep_sim_times_num, ret_sim_times_num)
                  where we return tuples instead of single values.

        :param option_start_date: start date of departure Air optionality
                                  - you can choose flights between these date and the option_end_date
        :param option_end_date: end date of departure Air Optionality - similar to start date.
        :param option_ret_start_date: start date of option return optionality,
                                      you can choose return flights starting from this date
        :param option_ret_end_date: end date of option return optionality, in conjunction w/ option_ret_start_date
        """

        if option_start_date or option_end_date:
            dep_sim_times_num = self.construct_sim_times( option_start_date
                                                        , option_end_date
                                                        , self.mkt_date
                                                        , simplify_compute = self.__simplify_compute )
        else:
            dep_flights = self.flights if not self.return_flight else self.flights[0]
            dep_sim_times_num = self.construct_sim_times( self.mkt_date
                                                        , min([dep_time for _, dep_time, _ in dep_flights])
                                                        , self.mkt_date
                                                        , simplify_compute=self.__simplify_compute)

        # all simulation times
        if self.return_flight:
            if option_ret_start_date or option_ret_end_date:
                ret_sim_times_num = self.construct_sim_times( option_ret_start_date
                                                            , option_ret_end_date
                                                            , self.mkt_date
                                                            , simplify_compute = self.__simplify_compute )
            else:
                ret_sim_times_num = self.construct_sim_times( self.mkt_date
                                                            , min([dep_time for _, dep_time, _ in self.flights[1]])
                                                            , self.mkt_date
                                                            , simplify_compute = self.__simplify_compute )

            return dep_sim_times_num, ret_sim_times_num

        return dep_sim_times_num

    def PV( self
          , option_start_date     : Optional[datetime.date] = None
          , option_end_date       : Optional[datetime.date] = None
          , option_ret_start_date : Optional[datetime.date] = None
          , option_ret_end_date   : Optional[datetime.date] = None
          , option_maturities     : Optional[List[datetime.date]] = None
          , nb_sim                : int                        = 1000
          , dcf                   : float                      = 365.25
          , cuda_ind              : bool                       = False ) -> float:
        """ Computes the value of the option for obtained flights in self.__flights
        If none of the inputs provided, use the default ones.

        All dates in the params are in datetime.date format
        :param option_start_date: start date of the option (default: market date, self.mkt_date)
        :param option_end_date: end date of the option (default: maturity of the first contract.)
        :param option_ret_start_date: maturity for returning option, if any; default: market date
        :param option_ret_end_date: maturity of the returning flights, default: earliest return flight date.
        :param option_maturities: list of maturities for which the option to be computed. If not None,
                                  all other values are overridden.
        :param nb_sim: number of simulations to use for computation.
        :param dcf: day count factor
        :param cuda_ind: indicator whether to use cuda
        """

        if not self.return_flight:
            if not self.flights:  # one-way flight
                return 0.

        else:  # return flights
            dep_flights, ret_flights = self.flights
            if (not dep_flights) or (not ret_flights):
                return 0.

        # extract forwards, etc. (sim_times might be overwritten below)
        sim_times = self.__dep_ret_sim_times_num(option_start_date, option_end_date, option_ret_start_date, option_ret_end_date)

        if option_maturities:  # special case
            # mapping between numerical maturities and datetime.date option_maturities
            sim_times = {(maturity_date - self.mkt_date).days / dcf: maturity_date
                         for maturity_date in option_maturities}

            sim_times_num  = list(sim_times.keys())  # numerical times
            sim_time_maturities = sim_times_num if not self.return_flight else (sim_times_num, sim_times_num)

        option_value = self.air_option_with_markup(sim_times if not option_maturities else sim_time_maturities  # simulation times
                                                   , self.K
                                                   , self.__rho
                                                   , nb_sim    = nb_sim
                                                   , cuda_ind      = cuda_ind
                                                   , underlyer     = self.__underlyer
                                                   , keep_all_sims = False if not option_maturities else True)

        if not option_maturities:
            return option_value

        return {sim_times[sim_time_num]: option_for_sim_time
                for sim_time_num, option_for_sim_time in option_value.items()
                if sim_time_num in sim_times }

    def __find_flight_by_number(self, flight_nb : str):
        """ Finds the flight number in self.flights by flight_nb

        :param flight_nb: flight number, e.g. 'UA71'
        :returns: index in the self.flights for one-way flight, or (0,1), index for return flight.
        """

        if not self.return_flight:
            return [fnb for _, _, fnb in self.flights].index(flight_nb)

        # return flight
        depart_nbs = [fnb for _, _, fnb in self.flights[0]]

        if flight_nb in depart_nbs:
            return 0, depart_nbs.index(flight_nb)

        return 1, [fnb for _, _, fnb in self.flights[1]].index(flight_nb)

    @functools.lru_cache(maxsize=128)
    def PV01( self
            , option_start_date     : Optional[datetime.date] = None
            , option_end_date       : Optional[datetime.date] = None
            , option_ret_start_date : Optional[datetime.date] = None
            , option_ret_end_date   : Optional[datetime.date] = None
            , nb_sim                : int                        = 10000
            , dcf                   : float                      = 365.25
            , bump_value            : float                      = 0.01 ) -> DeltaDict:
        """ Cached version of PV01 function. All parameters are the same.

        All dates in the params are in datetime.date format
        :param option_start_date: start date of the option (default: market date, self.mkt_date)
        :param option_end_date: end date of the option (default: maturity of the first contract.)
        :param option_ret_start_date: maturity for returning option, if any; default: market date
        :param option_ret_end_date: maturity of the returning flights, default: earliest return flight date.
        :param dcf: day count factor
        :param bump_value: value by which to bump forward ticket prices to compute the PV01
        """

        delta_dict = {}
        pv = self.PV( option_start_date     = option_start_date
                    , option_end_date       = option_end_date
                    , option_ret_start_date = option_ret_start_date
                    , option_ret_end_date   = option_ret_end_date
                    , nb_sim                = nb_sim
                    , dcf                   = dcf)  # original PV

        for flight_idx, (flight_value, flight_date, flight_nb) in enumerate(self.outbound_flights):
            self.outbound_flights[flight_idx] = (flight_value + bump_value, flight_date, flight_nb)

            new_value = self.PV( option_start_date     = option_start_date
                                 , option_end_date       = option_end_date
                                 , option_ret_start_date = option_ret_start_date
                                 , option_ret_end_date   = option_ret_end_date
                                 , nb_sim                = nb_sim
                                 , dcf                   = dcf)

            delta_dict[flight_nb] = (new_value -pv) / bump_value

            # setting the flight element back, after it was bumped above.
            self.outbound_flights[flight_idx] = (flight_value, flight_date, flight_nb)

        if not self.return_flight:
            return DeltaDict(delta_dict)

        # return flights
        for flight_idx, (flight_value, flight_date, flight_nb) in enumerate(self.inbound_flights):  # enumerate(self.flights[1]):
            new_flight_elt = (flight_value + bump_value, flight_date, flight_nb)  # bumping the flight element.
            self.inbound_flights[flight_idx] = new_flight_elt

            delta_diff = self.PV( option_start_date     = option_start_date
                                , option_end_date       = option_end_date
                                , option_ret_start_date = option_ret_start_date
                                , option_ret_end_date   = option_ret_end_date
                                , nb_sim                = nb_sim
                                , dcf                   = dcf) - pv

            delta_dict[flight_nb] = delta_diff / bump_value

            # setting the flight element back, after it was bumped above.
            self.inbound_flights[flight_idx] = (flight_value, flight_date, flight_nb)

        return DeltaDict(delta_dict)


    def air_option_with_markup(self
                               , sim_times     : Union[np.array, Tuple[np.array, np.array]]
                               , K             : float
                               , rho           : Union[float, np.ndarray, Tuple[np.ndarray, np.ndarray]]
                               , nb_sim        : int  = 10000
                               , cuda_ind      : bool = False
                               , underlyer     : str  = 'n'
                               , keep_all_sims : bool = False ):
        """ Computes the value of the option sequentially, in order to minimize memory footprint.

        :param sim_times: simulation times of for flight tickets; or a tuple for (departure, return tickets)
        :param K:         strike of the option
        :param rho:       correlation matrix for flight tickets, or a tuple of matrices for departure, return tickets
        :param nb_sim:    number of simulations
        :param cuda_ind:  whether to use cuda; True or False
        :param underlyer: which model to use - lognormal or normal ('ln' or 'n') - SO FAR ONLY NORMAL IS SUPPORTED.
        :param keep_all_sims: keeps all the simulations for sim_times and computes the option for each simulated date in sim_times
        """

        opt_val_final = self.air_option(sim_times
                                        , K
                                        , nb_sim    = nb_sim
                                        , rho       = rho
                                        , cuda_ind  = cuda_ind
                                        , underlyer = underlyer
                                        , keep_all_sims = keep_all_sims)

        logger.debug('Actual option value: {0}'.format(opt_val_final))

        # markups to the option value
        F_v = self._F_v
        percentage_markup = reserves + tax_rate
        F_v_max = np.max(F_v) if type(F_v) is not tuple else max(np.max(F_v[0]), np.max(F_v[1]))

        # minimal payoff
        min_payoff = max(MIN_PRICE, F_v_max / ref_base_F * MIN_PRICE)

        if not keep_all_sims:
            return max(min_payoff, (1. + percentage_markup) * opt_val_final)

        # keep_all_sims = True case
        return {sim_time: max(min_payoff, (1. + percentage_markup) * opt_val_final_at_time)
                for sim_time, opt_val_final_at_time in opt_val_final.items() }

    def _air_option_sims(self
                         , sim_times : Union[np.array, Tuple[np.array, np.array]]
                         , nb_sim        : int   = 1000
                         , rho           : float = 0.9
                         , cuda_ind      : bool  = False
                         , underlyer     : str   = 'n'
                         , keep_all_sims : bool = False):
        """ Only simulations used for air options.

        :param sim_times: simulation list, same as s_v
        :param nb_sim: number of simulations
        :param rho: correlation parameter, used only for now
        :param cuda_ind: indicator to use cuda
        :param underlyer: which model does the underlyer follow (normal 'n', log-normal 'ln')
        :param keep_all_sims: keeps all the simulations for sim_times
        """

        return_flight_ind = isinstance(self._F_v, tuple)

        F_v = self._F_v
        if return_flight_ind:
            # correlation matrix for departing, returning flights
            dep_flights, ret_flights = F_v
            rho_m = ( corr_hyp_sec_mat(rho, range(len(dep_flights)))
                    , corr_hyp_sec_mat(rho, range(len(ret_flights))) )

        else:  # only outgoing flight
            rho_m = corr_hyp_sec_mat(rho, range(len(F_v)))

        # which monte-carlo method to use.
        mc_used = self.mc_mult_steps if not return_flight_ind else self.mc_mult_steps_ret

        return mc_used( F_v
                      , self._s_v
                      , self._d_v
                      , sim_times
                      , rho_m
                      , self._F_mat_v
                      , nb_sim   = nb_sim
                      , model    = underlyer
                      , keep_all_sims= keep_all_sims)

    @functools.lru_cache
    def _generate_rn( self
                      , T_curr : float
                      , rho_m : Tuple[float]  # np.ndarray
                      , nb_fwds : int
                      , nb_sim : int ):

        if nb_fwds == 1:
            return np.random.normal(size=(1, nb_sim))

        rho_m_np = np.array(rho_m)

        return np.random.multivariate_normal(np.zeros(nb_fwds), rho_m_np, size=nb_sim)  # if not cuda_ind else mn_gpu(0, rho_m, size=nb_sim)

    def mc_mult_steps( self
                       , F_v     : [List, np.array]
                       , s_v     : np.array
                       , d_v     : np.array
                       , T_l     : np.array
                       , rho_m   : np.array
                       , T_v_exp: np.array
                       , nb_sim  = 1000
                       , model    = 'n'
                       , F_ret    = None
                       , keep_all_sims = False  ):  # -> [np.array, Dict[np.array]]:
        """ Multi-step monte-carlo integration of the ticket prices for one way Air options.

        :param F_v: list of forward values
        :param s_v: (vector of) volatility of the forward ticket process.
        :param d_v: drift of the forward ticket process.
        :param T_l: time points at which all F_v should be simulated
        :param rho_m: correlation matrix (2-dimensional array)
        :param T_v_exp: expiry of the forward contracts, maturity of aiplane tickets
        :param nb_sim: number of simulations
        :param model: model used for simulation, 'ln' for log-normal, or 'n' for normal
        :param F_ret: a simulated list of return values - size (nb_sim, 1)
        :param keep_all_sims: keeps all simulations for each sim_times
        :returns: matrix of simulation values in the shape [simulation, fwd] if keep_all_sims = False,
              or dictionary, where keys are simulation times and values are simulations for those times.
        """

        one_step_model = ln_step if model == 'ln' else normal_step

        T_l_extend = AirOptionFlights.add_zero_to_Tl(T_l)  # add 0 to simulation times if not already there
        T_l_diff   = np.diff(T_l_extend)
        nb_fwds    = len(F_v)

        F_sim = np.empty((nb_sim, nb_fwds))  # if not cuda_ind else gpa.empty((nb_sim, nb_fwds), np.double)
        F_sim[:, :] = F_v
        F_prev = F_sim  # previous simulations

        # also number of departure flights., nb_sim = nb. of simulations of those flights, also columns in F_sim
        nb_sim, nb_fwds = F_sim.shape  # nb of forward contracts, also nb of rows in F_sim

        for (T_diff, T_curr, T_prev) in zip(T_l_diff, T_l_extend[1:], T_l_extend[:-1]):  # T_diff = difference, T_curr: current time  # before: add_zero_to_Tl(T_l_diff)[:-1]
            ttm_used = np.array(T_v_exp) - T_curr
            s_v_used, d_v_used = vol_drift_vec(T_curr, T_diff, s_v, d_v, ttm_used)

            if nb_fwds == 1:
                rn_sim_l = self._generate_rn(T_curr, (), 1, nb_sim)  #    np.random.normal(size=(1, nb_sim))
            else:  # TODO: THIS LINE BELOW SHOULD BE IMPROVED
                rn_sim_l = self._generate_rn(T_curr, tuple([tuple(x) for x in rho_m]), nb_fwds, nb_sim)  #   np.random.multivariate_normal(np.zeros(nb_fwds), rho_m, size=nb_sim)  # if not cuda_ind else mn_gpu(0, rho_m, size=nb_sim)

            F_sim_next = one_step_model( F_sim if not keep_all_sims else F_prev
                                         , T_diff
                                         , s_v_used
                                         , d_v_used
                                         , rn_sim_l )

            if F_ret is None:  # no return flight given
                if not keep_all_sims:
                    F_sim = np.maximum(F_sim_next, F_sim)
                else:
                    yield T_curr, np.maximum(F_sim_next, F_prev)
                    F_prev = F_sim_next

            else:  # return flights
                F_sim_next_ret = F_sim_next + F_ret  # F_ret is already maximized over flights

                if not keep_all_sims:
                    F_sim = np.maximum(F_sim_next_ret, F_sim)
                else:
                    yield T_curr, np.maximum(F_sim_next_ret, F_prev)
                    F_prev = F_sim_next_ret

        if not keep_all_sims:
            yield T_curr, F_sim  # last value in T_curr

    @staticmethod
    def add_zero_to_Tl(T_list : List[float]) -> np.array:
        """ Prepends a zero to sim_times if sim_times doesnt already have it.

        :param T_list:  list of simulation times
        :returns:       array with first element as 0, if it isnt already zero. potential zero added
        """

        if T_list[0] != 0.:
            T_l_local = np.zeros(len(T_list) + 1)
            T_l_local[1:] = T_list
            return T_l_local

        return T_list

    def mc_mult_steps_ret( self
                           , F_v     : Tuple[np.array, np.array]
                           , s_v     : Tuple[np.array, np.array]
                           , d_v     : Tuple[np.array, np.array]
                           , T_l     : Tuple[List[float], List[float]]
                           , rho_m   : Tuple[np.array, np.array]
                           , T_v_exp : Tuple[np.array, np.array]
                           , nb_sim        = 1000
                           , model         = 'n'
                           , keep_all_sims = False ):
        """ Simulates ticket prices for a return flight.

        :param F_v: tuple of list of forward values, first for departure, second for return (arrays in tuple are 1-dim)
        :param s_v: tuple of list of vols for tickets, first departure, then return
        :param d_v: drift of the process, drift
                  d_v[i] is a function of (ttm, params)
        :param T_l: list of time points at which all F_v should be simulated
        :param rho_m: correlation matrix (2-dim array)
        :param nb_sim: number of sims
        :param T_v_exp: expiry of forward contracts, expiry in numerical terms.
        :param model: model 'ln' or 'n' for normal
        :param keep_all_sims: whether to keep the simulations
        :returns: matrix [time_step, simulation, fwd] or ticket prices
        """

        F_v_dep,     F_v_ret     = F_v
        s_v_dep,     s_v_ret     = s_v
        d_v_dep,     d_v_ret     = d_v
        T_l_dep,     T_l_ret     = AirOptionFlights.add_zero_to_Tl(T_l[0]), AirOptionFlights.add_zero_to_Tl(T_l[1])
        T_v_exp_dep, T_v_exp_ret = T_v_exp  # expiry values
        rho_m_dep,   rho_m_ret   = rho_m

        # mc_mult_steps is a generator and has exactly _0_ elements for keep_all_sims=False
        _, F_ret_realized = list(self.mc_mult_steps( F_v_ret
                                                     , s_v_ret
                                                     , d_v_ret
                                                     , T_l_ret
                                                     , rho_m_ret
                                                     , T_v_exp_ret
                                                     , nb_sim   = nb_sim
                                                     , model    = model ))[0]

        return self.mc_mult_steps( F_v_dep
                                   , s_v_dep
                                   , d_v_dep
                                   , T_l_dep
                                   , rho_m_dep
                                   , T_v_exp_dep
                                   , nb_sim   = nb_sim
                                   , model    = model
                                   , F_ret    = np.amax( F_ret_realized, axis = 1 ).reshape((nb_sim, 1))  # simulations in columns
                                   , keep_all_sims=keep_all_sims)

    def air_option(self
                   , sim_times : Union[np.array, Tuple[np.array, np.array]]
                   , K         : float
                   , nb_sim    = 1000
                   , rho       = 0.9
                   , cuda_ind  = False
                   , underlyer ='n'
                   , keep_all_sims = False) -> Union[float, Dict[str, float]]:
        """ Computes the value of the air option with low memory impact.

        :param sim_times: simulation list, same as s_v
        :param K: strike price
        :param nb_sim: simulation number
        :param rho: correlation parameter, used only for now
        :param cuda_ind: indicator to use cuda
        :param underlyer: which model does the underlyer follow (normal 'n', log-normal 'ln')
        :param keep_all_sims: keeps all the simulations for sim_times
        """

        F_max = self._air_option_sims(sim_times
                                      , nb_sim    = nb_sim
                                      , rho       = rho
                                      , cuda_ind  = cuda_ind
                                      , underlyer = underlyer
                                      , keep_all_sims = keep_all_sims)

        # final option payoff
        if not keep_all_sims:
            # realize a generator
            _, F_sim_realized = list(F_max)[0]
            return np.mean(np.maximum (np.amax(F_sim_realized, axis=0) - K, 0.))

            # cuda result
            # return np.mean(gpa.maximum(cuda_ops.amax_gpu_0(F_sim_realized) - K, 0.))

        # keep all simulation case
        return {sim_time: np.mean(np.maximum (np.amax(F_max_at_time, axis=0) - K, 0.))
                for sim_time, F_max_at_time in F_max}

    @staticmethod
    def compute_date_by_fraction( date_today : datetime.date
                                , date_final : datetime.date
                                , fract    : int
                                , total_fraction : int ) -> datetime.date:
        """
        Computes the date between dt_today and dt_final where the days between
        dt_today is the fract of dates between dt_today and dt_final

        :param date_today: "today's" date in datetime.date format
        :param date_final: final date that one considers for excersing the option
        :param fract: the fraction of the days between dt_today and dt_final (usually 3)
        :param total_fraction: total number of options that one considers (usually 3)
        :returns: outbound date fract/total_fraction between dt_today and dt_final
        """

        # fraction needs to be an integer
        # - 3 ... no change in the last 3 days
        return date_today + datetime.timedelta(days= (date_final - date_today).days * fract/total_fraction - 3)


def main():
    """ Example usage of some of the functions.
    """

    #ao1 = AirOptionFlights.from_db(datetime.date(2016, 1, 1), 1)
    # print(ao1.PV())
    pass
