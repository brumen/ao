# air option computation file
import config
import datetime
import numpy as np
import logging
import functools

from typing import List, Tuple

if config.CUDA_PRESENT:
    import cuda_ops
    import pycuda.gpuarray as gpa

import vols.vols as vols
import mc
import ds
import ao_codes
from air_flights import get_flight_data, find_minmax_flight_subset
from ao_codes    import MIN_PRICE

from ao_params import get_drift_vol_from_db

logger = logging.getLogger(__name__)


class AirOptionFlights:
    """ Handles the air option for a particular set of flights.

    """

    def __init__( self
                , mkt_date : datetime.date
                , flights               = None
                , option_start_date     = None
                , option_end_date       = None
                , option_ret_start_date = None
                , option_ret_end_date   = None
                , cuda_ind             = False
                , K                    = 1600.
                , nb_sim               = 10000
                , rho                  = 0.95
                , simplify_compute     = 'take_last_only'
                , underlyer            = 'n'
                , correct_drift        = True ):
        """
        Computes the air option for the flights

        :param mkt_date: market date
        :param flights: flights to compute the air option over.
                        pairs of (flight forward value, flight_date(forward maturity date), flight_nb )
        :type flights: [(double, datetime.date, str)] or tuple([(double, datetime.date, str)], [(double, datetime.date, str)])

        :param option_start_date:       the date when you can start changing the outbound flight
        :type option_start_date:        datetime.date
        :param option_end_date:         the date when you stop changing the outbound flight
        :type option_end_date:          datetime.date
        :param option_ret_start_date:   the date when you can start changing the inbound flight
        :type option_ret_start_date:    datetime.date
        :param option_ret_end_date:     the date when you stop changing the outbound flight
        :type option_ret_end_date:      datetime.date
        :param K:                       option strike
        :type K:                        double
        :param cuda_ind:                whether to use cuda for computation
        :type cuda_ind:                 bool
        :param nb_sim:                  number of simulations
        :type nb_sim:                   int
        :param rho:                     correlation between flights parameter
        :type rho:                      double
        :param simplify_compute:        simplifies the computation in that it only simulates the last simulation date,
                                          options are: "take_last_only", "all_sim_dates"
        :type simplify_compute:         str
        """

        self.mkt_date  = mkt_date
        self.__flights = flights
        self.return_flight = True if isinstance(self.__flights, tuple) else False  # return flight indicator

        self.__K = K
        self.__nb_sim = nb_sim
        self.__rho = rho
        self.__cuda_ind = cuda_ind
        self.__underlyer = underlyer
        self.__correct_drift = correct_drift

        self.__simplify_compute     = simplify_compute

        # option date setup default, could be something or None
        self.__option_start_date     = option_start_date
        self.__option_end_date       = option_end_date
        self.__option_ret_start_date = option_ret_start_date
        self.__option_ret_end_date   = option_ret_end_date

        # caching variables
        self.__recompute_option_value = True  # indicator whether the option should be recomputed
        self.__option_value = None

    @property
    def cuda_ind(self):
        return self.__cuda_ind

    @cuda_ind.setter
    def cuda_ind(self, new_cuda_ind):
        self.__cuda_ind = new_cuda_ind

    @property
    def flights(self):
        return self.__flights

    @flights.setter
    def flights(self, new_flights):
        self.__recompute_option_value = True  # recompute everything
        self.__flights = new_flights

    @property
    def option_start_date(self) -> datetime.date :
        """ Default option start date if none provided, else option_start_date, if already set.

        """

        # TODO: THERE SHOULD BE SOME BOUNDARY VALUES CHECKING.
        if not self.__option_start_date:
            self.__option_start_date = self.mkt_date  # default option start date

        return self.__option_start_date

    @option_start_date.setter
    def option_start_date(self, new_start_date : datetime.date ):
        """ Setting a new option start date.

        :param new_start_date: new option start date

        """

        if new_start_date != self.__option_start_date:
            self.__option_start_date      = new_start_date
            self.__recompute_option_value = True

    @property
    def option_end_date(self):
        """ Option end date, either set explicitly, or defaults to the first of the flight.

        """

        if not self.__option_end_date:
            # first elt of self._flights are departing flights
            dep_flights = self.flights if not self.return_flight else self.flights[0]
            self.__option_end_date = min([dep_time for _, dep_time, _ in dep_flights])

        return self.__option_end_date

    @option_end_date.setter
    def option_end_date(self, new_end_date):

        if new_end_date != self.__option_end_date:
            self.__option_end_date = new_end_date
            self.__recompute_option_value = True

    @property
    def option_ret_start_date(self) -> datetime.date :
        """ Default option start date if none provided, else option_start_date, if already set.

        """

        # TODO: THERE SHOULD BE SOME BOUNDARY VALUES CHECKING.
        if not self.__option_ret_start_date:
            self.__option_ret_start_date = self.mkt_date  # default option start date

        return self.__option_ret_start_date

    @option_ret_start_date.setter
    def option_ret_start_date(self, new_start_date : datetime.date ):
        """ Setting a new option start date.

        :param new_start_date: new option start date
        """

        if new_start_date != self.__option_ret_start_date:
            self.__option_ret_start_date = new_start_date
            self.__recompute_option_value = True

    @property
    def option_ret_end_date(self):
        """ Option end date, either set explicitly, or defaults to the first of the flight

        """

        if not self.__option_ret_end_date:
            # first elt of self._flights are departing flights
            self.__option_ret_end_date = min([dep_time for _, dep_time, _ in self.flights[1]])

        return self.__option_ret_end_date

    @option_ret_end_date.setter
    def option_ret_end_date(self, new_end_date):

        if new_end_date != self.__option_ret_end_date:
            self.__option_ret_end_date = new_end_date
            self.__recompute_option_value = True

    @staticmethod
    def construct_sim_times( date_start    : datetime.date
                           , date_end      : datetime.date
                           , date_today_dt : datetime.date
                           , simplify_compute = 'take_last_only'
                           , dcf              = 365.) -> List[datetime.date]:
        """ Constructs the simulation times used for air options.

        :param date_start: date start of simulated times
        :param date_end: date end of simulated times
        :param date_today_dt: reference date, mostly today's date
        :param dcf: day count factor
        :type dcf: double
        :returns: list of simulated dates
        """

        T_l = ds.construct_date_range(date_start, date_end)  # in date format

        if simplify_compute == 'all_sim_dates':
            return [(date_sim - date_today_dt).days / dcf for date_sim in T_l]

        # elif simplify_compute == 'take_last_only':
        return [(T_l[-1] - date_today_dt).days / dcf]

    @functools.lru_cache(maxsize=128)
    def _drift_vol_for_flight(self, flight_nb: Tuple[datetime.date, str]) -> Tuple[float, float]:
        """ Gets drift and vol for flights. It caches it, so that drift and vol are not regetting it.

        """

        return (0.3, 0.3)

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

    def _get_origin_dest_carrier_from_flight(self, flight_id) -> Tuple[str, str, str]:
        """ Gets origin, destination, carrier from flight_id, e.g. 'UA70' is 'SFO', 'EWR', 'UA'

        :param flight_id: flight id
        :returns: origin airport, destination airport, carrier
        """

        # TODO: IMPLEMENT HERE IF YOU CAN!!!
        return 'SFO', 'EWR', 'UA'

    def __extract_prices_maturities(self, flights : List[Tuple[float, datetime.date, str]], dcf = 365.25):
        """ Extracts flight prices, maturities, and obtains drifts and volatilities from flights.

        :param flights: flights list of (flight_price, flight_date, flight_nb)
        :param dcf: day-count convention
        """

        F_v, F_mat, s_v, d_v = [], [], [], []

        # TODO: THIS CAN BE REWRITTEN BETTER USING ZIP
        for (F_dep_value, F_dep_maturity, flight_nb) in flights:
            F_v.append(F_dep_value)
            F_mat.append((F_dep_maturity - self.mkt_date).days/dcf)  # maturity has to be in numeric terms
            s_v.append(self._vol_for_flight  ((F_dep_maturity, flight_nb)))
            d_v.append(self._drift_for_flight((F_dep_maturity, flight_nb)))

        return F_v, F_mat, s_v, d_v

    def __extract_prices_maturities_all(self):
        """ Same as extract_prices_maturities, just that it considers both departing, returning flights

        """

        # all simulation times
        T_l_dep_num = AirOptionFlights.construct_sim_times( self.option_start_date
                                                          , self.option_end_date
                                                          , self.mkt_date
                                                          , simplify_compute = self.__simplify_compute)

        if self.return_flight:
            T_l_ret_num = AirOptionFlights.construct_sim_times( self.option_ret_start_date
                                                              , self.option_ret_end_date
                                                              , self.mkt_date
                                                              , simplify_compute = self.__simplify_compute )

        if self.return_flight:  # return flights
            departing_flights, returning_flights = self.flights
            F_v_dep, F_mat_dep, s_v_dep, d_v_dep = self.__extract_prices_maturities(departing_flights)
            F_v_ret, F_mat_ret, s_v_ret, d_v_ret = self.__extract_prices_maturities(returning_flights)

            return (F_v_dep, F_v_ret), (F_mat_dep, F_mat_ret), (s_v_dep, s_v_ret), (d_v_dep, d_v_ret), (T_l_dep_num, T_l_ret_num)

        # one-way flights
        F_v, F_mat, s_v, d_v = self.__extract_prices_maturities(self.flights)

        return F_v, F_mat, s_v, d_v, T_l_dep_num

    def __call__( self
                , option_start_date     = None
                , option_end_date       = None
                , option_ret_start_date = None
                , option_ret_end_date   = None ):
        """ Computes the value of the option for obtained flights in self.__flights
        If none of the inputs provided, use the default ones.

        """

        self.option_start_date     = option_start_date
        self.option_end_date       = option_end_date
        self.option_ret_start_date = option_ret_start_date
        self.option_ret_end_date   = option_ret_end_date

        if not self.__recompute_option_value:
            return self.__option_value

        # recompute option value
        F_v, F_mat, s_v, d_v, T_l = self.__extract_prices_maturities_all()

        self.__option_value = self.__class__.compute_option_raw( F_v
                                                               , s_v
                                                               , d_v
                                                               , T_l
                                                               , F_mat
                                                               , self.__K
                                                               , self.__rho
                                                               , nb_sim    = self.__nb_sim
                                                               , cuda_ind  = self.cuda_ind
                                                               , underlyer = self.__underlyer)

        self.__recompute_option_value = False

        return self.__option_value

    def option_range(self, option_maturities : List[datetime.date], dcf = 365.25):
        """ Constructs a series of option prices for different maturities.

        :param option_maturities: maturities for which options should be computed
        :param dcf: day count factor
        """

        F_v, F_mat, s_v, d_v, _ = self.__extract_prices_maturities_all()

        # mapping between numerical maturities and option_maturities
        sim_times = {(maturity_date - self.mkt_date).days/dcf : maturity_date
                     for maturity_date in option_maturities }

        # TODO: THIS SHOULD BE BETTER DONE
        sim_times_num = list(sim_times.keys())  # numerical times

        options_for_num_times = self.__class__.compute_option_raw( F_v
                                                , s_v
                                                , d_v
                                                , sim_times_num if not self.return_flight else (sim_times_num, sim_times_num)
                                                , F_mat
                                                , self.__K
                                                , self.__rho
                                                , nb_sim        = self.__nb_sim
                                                , cuda_ind      = self.cuda_ind
                                                , underlyer     = self.__underlyer
                                                , keep_all_sims = True )

        return {sim_times[sim_time_num]: option_for_sim_time
                for sim_time_num, option_for_sim_time in options_for_num_times.items()
                if sim_time_num in sim_times }

    @staticmethod
    def compute_option_raw( F_v
                          , s_v
                          , d_v
                          , T_l_num
                          , T_mat_num
                          , K
                          , rho
                          , nb_sim    = 10000
                          , cuda_ind  = False
                          , underlyer = 'n'
                          , keep_all_sims = False ):
        """
        Computes the value of the option sequentially, in order to minimize memory footprint

        :param F_v:       vector of tickets for one-way flights, tuple for return flights
        :type F_v:        np.array or tuple(np.array, np.array)
        :param s_v:       vector of vols for one-way, the same size as F_v, or tuple for return flights
        :type s_v:        np.array or tuple(np.array, np.array)
        :param d_v:       vector of drifts for one-way, the same size as F_v, or tuple for return flights
        :type d_v:        np.array or tuple(np.array, np.array)
        :param T_l_num:   simulation times of for tickets; or a tuple for departure, return tickets
        :type T_l_num:    np.array 1-dimensional; or tuple (np.array, np.array)
        :param T_mat_num: maturity of tickets TODO
        :type T_mat_num:
        :param K:         strike of the option
        :type K:          double
        :param rho:       correlation matrix for flight tickets, or a tuple of matrices for departure, return tickets
        :type rho:        np.array 2 dimensional; or a tuple of two such matrices
        :param nb_sim:    number of simulations
        :type nb_sim:     integer
        :param cuda_ind:  whether to use cuda; True or False
        :param underlyer: which model to use - lognormal or normal ('ln' or 'n') - SO FAR ONLY NORMAL IS SUPPORTED.
        :type underlyer:  string
        :param keep_all_sims: keeps all the simulations for T_l
        :returns:
        :rtype:
        """

        opt_val_final = AirOptionFlights.air_option( F_v
                                                   , s_v
                                                   , d_v
                                                   , T_l_num
                                                   , T_mat_num
                                                   , K
                                                   , nb_sim    = nb_sim
                                                   , rho       = rho
                                                   , cuda_ind  = cuda_ind
                                                   , underlyer = underlyer
                                                   , keep_all_sims = keep_all_sims )

        # markups to the option value
        percentage_markup = ao_codes.reserves + ao_codes.tax_rate
        F_v_max = np.max(F_v) if type(F_v) is not tuple else  max(np.max(F_v[0]), np.max(F_v[1]))

        # minimal payoff
        min_payoff = max(MIN_PRICE, F_v_max / ao_codes.ref_base_F * MIN_PRICE)

        if not keep_all_sims:
            return max(min_payoff, (1. + percentage_markup) * opt_val_final)

        # keep_all_sims = True case
        return {sim_time: max(min_payoff, (1. + percentage_markup) * opt_val_final_at_time)
                for sim_time, opt_val_final_at_time in opt_val_final.items() }

    @staticmethod
    def air_option( F_v
                  , s_v
                  , d_v
                  , T_l
                  , T_mat
                  , K
                  , nb_sim    = 1000
                  , rho       = 0.9
                  , cuda_ind  = False
                  , underlyer ='n'
                  , keep_all_sims = False):
        """ Computes the value of the air option with low memory impact.

        :param F_v: vector of forward prices, or a tuple (F_1_v, F_2_v) for return flights
        :type F_v: np.array or (np.array, np.array)
        :param s_v: vector of vols, or a tuple for return flights, similarly to F_v
        :type s_v: np.array or (np.array, np.array)
        :param d_v: functions that describe the drift of the forward (list form)
           d_v[i] is a function of (F_prev, ttm, time_step, params)
        :param T_l: simulation list, same as s_v
        :type T_l: np.array or (np.array, np.array)
        :param T_mat: maturity list
        :param K: strike price
        :param nb_sim: simulation number
        :param rho: correlation parameter, used only for now
        :param cuda_ind: indicator to use cuda
        :param underlyer: which model does the underlyer follow (normal 'n', log-normal 'ln')
        :param keep_all_sims: keeps all the simulations for T_l
        """

        return_flight_ind = isinstance(F_v, tuple)

        if return_flight_ind:
            # correlation matrix for departing, returning flights
            rho_m = ( vols.corr_hyp_sec_mat(rho, range(len(F_v[0])))
                    , vols.corr_hyp_sec_mat(rho, range(len(F_v[1]))) )

        else:  # only outgoing flight
            rho_m = vols.corr_hyp_sec_mat(rho, range(len(F_v)))

        mc_used = mc.mc_mult_steps if not return_flight_ind else mc.mc_mult_steps_ret

        F_max = mc_used( F_v
                       , s_v
                       , d_v
                       , T_l
                       , rho_m
                       , T_mat
                       , nb_sim   = nb_sim
                       , model    = underlyer
                       , cuda_ind = cuda_ind
                       , keep_all_sims= keep_all_sims )  # simulation of all flight prices

        # final option payoff
        if not keep_all_sims:
            if not cuda_ind:
                return np.mean(np.maximum (np.amax(F_max, axis=0) - K, 0.))

            # cuda result
            return np.mean(gpa.maximum(cuda_ops.amax_gpu_0(F_max) - K, 0.))

        # keep all simulation case
        if not cuda_ind:
            return {sim_time: np.mean(np.maximum (np.amax(F_max_at_time, axis=0) - K, 0.))
                    for sim_time, F_max_at_time in F_max.items()}

        # cuda result
        return {sim_time: np.mean(gpa.maximum(cuda_ops.amax_gpu_0(F_max_at_time) - K, 0.))
                for sim_time, F_max_at_time in F_max.items()}

    @staticmethod
    def compute_date_by_fraction( dt_today : datetime.date
                                , dt_final : datetime.date
                                , fract    : int
                                , total_fraction : int ) -> datetime.date:
        """
        Computes the date between dt_today and dt_final where the days between
        dt_today is the fract of dates between dt_today and dt_final

        :param dt_today:       "today's" date in datetime.date format
        :param dt_final:       final date that one considers for excersing the option
        :param fract:          the fraction of the days between dt_today and dt_final (usually 3)
        :param total_fraction: total number of options that one considers (usually 3)
        :returns:              outbound date fract/total_fraction between dt_today and dt_final
        """

        # fraction needs to be an integer
        # - 3 ... no change in the last 3 days
        return dt_today + datetime.timedelta(days= (dt_final - dt_today).days * fract/total_fraction - 3)


class AirOptionSkyScanner(AirOptionFlights):
    """
    Class for handling the air options for SkyScanner inputs.

    """

    def __init__( self
                , mkt_date
                , origin    = 'SFO'
                , dest      = 'EWR'
                # when can you change the option
                , option_start_date     = None
                , option_end_date       = None
                , option_ret_start_date = None
                , option_ret_end_date   = None
                # next 4 - when do the (changed) flights occur
                , outbound_date_start = None
                , outbound_date_end   = None
                , inbound_date_start  = None
                , inbound_date_end    = None
                , K          = 1600.
                , carrier    = 'UA'
                , nb_sim     = 10000
                , rho        = 0.95
                , adults     = 1
                , cabinclass = 'Economy'
                , cuda_ind   = False
                , simplify_compute = 'take_last_only'
                , underlyer        = 'n'
                , return_flight    = False
                , recompute_ind    = False
                , correct_drift    = True ):
        """
        Computes the air option from the data provided.

        :param origin: IATA code of the origin airport ('SFO')
        :param dest: IATA code of the destination airport ('EWR')
        :param option_start_date: the date when you can start changing the outbound flight
        :type option_start_date: datetime.date
        :param option_end_date: the date when you stop changing the outbound flight
        :type option_end_date: datetime.date
        :param option_ret_start_date: the date when you can start changing the inbound flight
        :type option_ret_start_date: datetime.date
        :param option_ret_end_date: the date when you stop changing the outbound flight
        :type option_ret_end_date: datetime.date
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

        super(AirOptionSkyScanner, self).__init__( mkt_date              = self.mkt_date
                                                 , flights               = list(self.get_flights())
                                                 , option_start_date     = option_start_date
                                                 , option_end_date       = option_end_date
                                                 , option_ret_start_date = option_ret_start_date
                                                 , option_ret_end_date   = option_ret_end_date
                                                 , cuda_ind              = cuda_ind
                                                 , rho                   = rho
                                                 , nb_sim                = nb_sim
                                                 , K                     = K
                                                 , simplify_compute      = simplify_compute
                                                 , underlyer             = underlyer )

    def get_flights(self):
        """ Returns the flights from SkyScanner.

        """
        return get_flight_data( origin_place        = self.__origin
                              , dest_place          = self.__dest
                              , outbound_date_start = self.__outbound_date_start
                              , outbound_date_end   = self.__outbound_date_end
                              , inbound_date_start  = self.__inbound_date_start
                              , inbound_date_end    = self.__inbound_date_end
                              , carrier             = self.__carrier
                              , cabinclass          = self.__cabinclass
                              , adults              = self.__adults
                              , return_flight       = self.__return_flight
                              , recompute_ind       = self.__recompute_ind
                              , correct_drift       = self.__correct_drift )

    @functools.lru_cache(maxsize=128)
    def _drift_vol_for_flight(self, flight_nb: Tuple[datetime.date, str]) -> Tuple[float, float]:
        """ Gets drift and vol for flights. It caches it, so that drift and vol are not regetting it.

        """

        dep_date, flight_id = flight_nb
        orig, dest, carrier = self._get_origin_dest_carrier_from_flight(flight_id)

        return get_drift_vol_from_db( dep_date
                                    , orig
                                    , dest
                                    , carrier
                                    , default_drift_vol = (500., 501.)
                                    , fwd_value         = None
                                    , db_host           = 'localhost' )


class AirOptionMock(AirOptionSkyScanner):
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
