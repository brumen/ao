# air option computation file
import config
import datetime
import numpy as np
import logging

from typing import List

if config.CUDA_PRESENT:
    import cuda_ops
    import pycuda.gpuarray as gpa

import vols.vols as vols
import mc
import ds
import ao_codes
from air_flights import get_flight_data, find_minmax_flight_subset


from ao_codes import MIN_PRICE


logger = logging.getLogger(__name__)


def construct_sim_times( date_start    : datetime.date
                       , date_end      : datetime.date
                       , date_today_dt : datetime.date
                       , simplify_compute
                       , dcf = 365. ) -> List[datetime.date] :
    """
    Constructs the simulation times used for air options.

    :param date_start:    date start
    :param date_end:      date end
    :param date_today_dt: date today
    :param dcf:           day count factor
    :type dcf:            double
    :returns:             list of simulated dates
    """

    T_l = ds.construct_date_range(date_start, date_end)  # in date format

    if simplify_compute == 'all_sim_dates':
        return [(date_sim - date_today_dt).days/dcf for date_sim in T_l]

    # elif simplify_compute == 'take_last_only':
    return [(T_l[-1] - date_today_dt).days/dcf]


class AirOptionFlights:
    """ Handles the air option for a particular set of flights.

    """

    def __init__( self
                , mkt_date : datetime.date
                , flights
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
                , price_by_range       = True
                , correct_drift        = True
                , compute_all          = True ):
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
        self._flights = flights
        self.return_flight = True if isinstance(self._flights, tuple) else False  # return flight indicator

        self.__K = K
        self.__nb_sim = nb_sim
        self.__rho = rho
        self.__cuda_ind = cuda_ind
        self.__underlyer = underlyer
        self.__price_by_range = price_by_range
        self.__correct_drift = correct_drift
        self.__compute_all = compute_all

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
            self.__option_start_date = new_start_date
            self.__recompute_option_value = True

    @property
    def option_end_date(self):
        """ Option end date, either set explicitly, or defaults to the first of the flight

        """

        if not self.__option_end_date:
            # first elt of self._flights are departing flights
            dep_flights = self._flights if not self.return_flight else self._flights[0]
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
            self.__option_ret_end_date = min([dep_time for _, dep_time, _ in self._flights[1]])

        return self.__option_ret_end_date

    @option_ret_end_date.setter
    def option_ret_end_date(self, new_end_date):

        if new_end_date != self.__option_ret_end_date:
            self.__option_ret_end_date = new_end_date
            self.__recompute_option_value = True

    def __getOutboundTL(self, outbound_date_start):
        """
        Ancilliary function for getting the outbound dates and simulation times.
        TODO: COMMENT HERE
        """

        return AirOption.compute_date_by_fraction( self.mkt_date
                                                 , outbound_date_start
                                                 , self.__complete_set_options - ri
                                                 , self.__complete_set_options) \
            , construct_sim_times(self.mkt_date
                                  , outbound_date_consid
                                  , datetime.date.today()
                                  , simplify_compute=self.__simplify_compute)

    def _drift_for_flight(self, flight_nb : str) -> float:
        """ Drift for flight, this method can be reimplemented.

        :param flight_nb: flight number
        """

        return 0.3

    def _vol_for_flight(self, flight_nb : str) -> float:
        """ Drift for flight, this method can be reimplemented.

        :param flight_nb: flight number
        """

        return 0.3

    def extract_prices_maturities(self, flights, dcf=365.25):
        """ Extracts flight prices, maturities, and obtains drifts and volatilities from flights.

        :param flights: flights list of (flight_price, flight_date, flight_nb)
        :type flights: list[(float, datetime.date, str)]
        :param dcf: day-count convention
        """

        F_v, F_mat, s_v, d_v = [], [], [], []

        # TODO: THIS CAN BE REWRITTEN BETTER USING ZIP
        for (F_dep_value, F_dep_maturity, flight_nb) in flights:
            F_v.append(F_dep_value)
            F_mat.append((F_dep_maturity - self.mkt_date).days/dcf)  # maturity has to be in numeric terms
            s_v.append(self._vol_for_flight(flight_nb))
            d_v.append(self._drift_for_flight(flight_nb))

        return F_v, F_mat, s_v, d_v

    def extract_prices_maturities_all(self):
        """ Same as extract_prices_maturities, just that it considers both departing, returning flights

        """

        # all simulation times
        T_l_dep_num = construct_sim_times(self.option_start_date
                                         , self.option_end_date
                                         , self.mkt_date
                                         , simplify_compute = self.__simplify_compute)

        if self.return_flight:
            T_l_ret_num = construct_sim_times( self.option_ret_start_date
                                             , self.option_ret_end_date
                                             , self.mkt_date
                                             , simplify_compute = self.__simplify_compute )

        if self.return_flight:  # return flights
            departing_flights, returning_flights = self._flights
            F_v_dep, F_mat_dep, s_v_dep, d_v_dep = self.extract_prices_maturities(departing_flights)
            F_v_ret, F_mat_ret, s_v_ret, d_v_ret = self.extract_prices_maturities(returning_flights)

            return (F_v_dep, F_v_ret), (F_mat_dep, F_mat_ret), (s_v_dep, s_v_ret), (d_v_dep, d_v_ret), (T_l_dep_num, T_l_ret_num)

        # one-way flights
        F_v, F_mat, s_v, d_v = self.extract_prices_maturities(self._flights)
        return F_v, F_mat, s_v, d_v, T_l_dep_num

    def __call__( self
                , option_start_date     = None
                , option_end_date       = None
                , option_ret_start_date = None
                , option_ret_end_date   = None ):
        """ Computes the value of the option for obtained flights in self.__flights
        If none of the inputs provided, use the default ones.

        """

        self.option_start_date = option_start_date
        self.option_end_date   = option_end_date
        self.option_ret_start_date = option_ret_start_date
        self.option_ret_end_date   = option_ret_end_date

        if not self.__recompute_option_value:
            return self.__option_value

        # recompute option value
        F_v, F_mat, s_v, d_v, T_l = self.extract_prices_maturities_all()

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

    # TODO: IMPROVE THIS METHOD LOOKS WEIRD
    def option_range(self, option_maturities : List[datetime.date]):
        """ Constructs a series of options for different maturities.

        :param option_maturities: maturities for which options should be computed
        """

        F_v, F_mat, s_v, d_v, _ = self.extract_prices_maturities_all()

        return self.__class__.compute_option_raw( F_v
                                                , s_v
                                                , d_v
                                                , option_maturities if not self.return_flight else (option_maturities, option_maturities)  # TODO: CHECK CHECK CHECK
                                                , F_mat
                                                , self.__K
                                                , self.__rho
                                                , nb_sim    = self.__nb_sim
                                                , cuda_ind  = self.cuda_ind
                                                , underlyer = self.__underlyer)

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
                          , underlyer = 'n' ):
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
        :type cuda_ind:   bool
        :param underlyer: which model to use - lognormal or normal ('ln' or 'n') - SO FAR ONLY NORMAL IS SUPPORTED.
        :type underlyer:  string; 'ln' or 'n'
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
                                  , underlyer = underlyer )

        # markups to the option value
        percentage_markup = ao_codes.reserves + ao_codes.tax_rate
        F_v_max = np.max(F_v) if type(F_v) is not tuple else  max(np.max(F_v[0]), np.max(F_v[1]))

        # minimal payoff
        min_payoff = max(MIN_PRICE, F_v_max / ao_codes.ref_base_F * MIN_PRICE)

        return max(min_payoff, (1. + percentage_markup) * opt_val_final)

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
                  , underlyer ='n' ):
        """
        Computes the value of the air option with low memory impact.

        :param F_v: vector of forward prices, or a tuple (F_1_v, F_2_v) for return flights
        :type F_v: np.array or (np.array, np.array)
        :param s_v: vector of vols, or a tuple for return flights, similarly to F_v
        :type s_v: np.array or (np.array, np.array)
        :param T_l: simulation list, same as s_v
        :type T_l: np.array or (np.array, np.array)
        :param T_mat: maturity list
        :param K: strike price
        :param d_v: functions that describe the drift of the forward (list form)
           d_v[i] is a function of (F_prev, ttm, time_step, params)
        :param nb_sim: simulation number
        :param rho: correlation parameter, used only for now
        :param cuda_ind: indicator to use cuda
        :param underlyer: which model does the underlyer follow (normal 'n', log-normal 'ln')
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
                       , cuda_ind = cuda_ind)  # simulation of all flight prices

        # final option payoff
        if not cuda_ind:
            return np.mean(np.maximum (np.amax(F_max, axis=0) - K, 0.))

        # cuda result
        return np.mean(gpa.maximum(cuda_ops.amax_gpu_0(F_max) - K, 0.))

    @staticmethod
    def compute_date_by_fraction( dt_today : datetime.date
                                , dt_final : datetime.date
                                , fract    : int
                                , total_fraction : int ) -> datetime.date :
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


class AirOption:
    """
    Class for handling the air options

    """

    def __init__( self
                , origin_place='SFO'
                , dest_place='EWR'
                , flights_include=None
                # when can you change the option
                , option_start_date=None
                , option_end_date=None
                , option_ret_start_date=None
                , option_ret_end_date=None
                # next 4 - when do the (changed) flights occur
                , outbound_date_start=None
                , outbound_date_end=None
                , inbound_date_start=None
                , inbound_date_end=None
                , K=1600.
                , carrier='UA'
                , nb_sim=10000
                , rho=0.95
                , adults = 1
                , cabinclass='Economy'
                , cuda_ind=False
                , simplify_compute='take_last_only'
                , underlyer='n'
                , price_by_range=True
                , return_flight=False
                , recompute_ind=False
                , correct_drift=True
                , compute_all=True
                , complete_set_options=3):
        """
        Computes the air option from the data provided.

        :param origin_place:            IATA code of the origin airport ('SFO')
        :type origin_place:             str
        :param dest_place:              IATA code of the destination airport ('EWR')
        :type dest_place:               str
        :param flights_include:         list of flights to include in pricing this option
        :type flights_include:          list of tuples # TODO: BE MORE PRECISE HERE
        :param option_start_date:       the date when you can start changing the outbound flight
        :type option_start_date:        datetime.date
        :param option_end_date:         the date when you stop changing the outbound flight
        :type option_end_date:          datetime.date
        :param option_ret_start_date:   the date when you can start changing the inbound flight
        :type option_ret_start_date:    datetime.date
        :param option_ret_end_date:     the date when you stop changing the outbound flight
        :type option_ret_end_date:      datetime.date
        :param outbound_date_start:     start date for outbound flights to change to
        :type outbound_date_start:      datetime.date
        :param outbound_date_end:       end date for outbound flights to change to
        :type outbound_date_end:        datetime.date
        :param inbound_date_start:      start date for inbound flights to change to
        :type inbound_date_start:       datetime.date
        :param inbound_date_end:        end date for inbound flights to change to
        :type inbound_date_end:         datetime.date
        :param K:                       option strike
        :type K:                        double
        :param carrier:                 IATA code of the carrier
        :type carrier:                  str
        :param nb_sim:                  number of simulations
        :type nb_sim:                   int
        :param rho:                     correlation between flights parameter
        :type rho:                      double
        :param adults:                  nb. of people on this ticket
        :type adults:                   int
        :param cabinclass:              class of flight ticket
        :type cabinclass:               str
        :param cuda_ind:                whether to use cuda for computation
        :type cuda_ind:                 bool
        :param simplify_compute:        simplifies the computation in that it only simulates the last simulation date
        :type simplify_compute:         str, options are: "take_last_only", "all_sim_dates"
        """

        flights = get_flight_data( flights_include     = flights_include
                                  , origin_place        = origin_place
                                  , dest_place          = dest_place
                                  , outbound_date_start = outbound_date_start
                                  , outbound_date_end   = outbound_date_end
                                  , inbound_date_start  = inbound_date_start
                                  , inbound_date_end    = inbound_date_end
                                  , carrier             = carrier
                                  , cabinclass=cabinclass
                                  , adults=adults
                                  , return_flight=return_flight
                                  , recompute_ind=recompute_ind
                                  , correct_drift=correct_drift )
                  # , nb_adults     = adults
                  # , return_flight = return_flight
                  # , cuda_ind      = cuda_ind
                  # , option_start_date = option_start_date
                  # , option_end_date   = option_end_date
                  # , option_ret_start_date = option_ret_start_date
                  # , option_ret_end_date   = option_ret_end_date )
