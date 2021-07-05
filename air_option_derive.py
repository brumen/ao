# air option computation file
import datetime
import numpy as np
import logging
import functools

from typing import List, Tuple, Optional

from ao.air_flights import get_flight_data
from ao.ao_params   import get_drift_vol_from_db
from ao.flight      import AOTrade, Flight, create_session
from ao.air_option  import AirOptionFlights

logging.basicConfig(filename='/tmp/air_option.log')
logger = logging.getLogger(__name__)
logger.setLevel('INFO')


class AOTradeException(Exception):
    pass


class AirOptionSkyScanner(AirOptionFlights):
    """ Class for handling the air options from SkyScanner inputs.
    """

    def __init__( self
                , mkt_date  : datetime.date
                , origin    = 'SFO'
                , dest      = 'EWR'
                # next 4 - when do the (changed) flights occur
                , outbound_date_start : Optional[datetime.date] = None
                , outbound_date_end   : Optional[datetime.date] = None
                , inbound_date_start  : Optional[datetime.date] = None
                , inbound_date_end    : Optional[datetime.date] = None
                , K                   : float = 1600.
                , carrier             : str   = 'UA'
                , rho                 : float = 0.95
                , adults              : int   = 1
                , cabinclass          : str   = 'Economy'
                , simplify_compute    : str   = 'take_last_only'
                , underlyer           : str   = 'n'
                , return_flight       : bool  = False
                , recompute_ind       : bool  = False
                , correct_drift       : bool  = True
                , db_host             : str   = 'localhost' ):
        """ Computes the air option from the data provided.

        :param origin: IATA code of the origin airport ('SFO')
        :param dest: IATA code of the destination airport ('EWR')
        :param outbound_date_start: start date for outbound flights to change to
        :param outbound_date_end: end date for outbound flights to change to
        :param inbound_date_start: start date for inbound flights to change to
        :param inbound_date_end: end date for inbound flights to change to
        :param K: option strike
        :param carrier: IATA code of the carrier
        :param rho: correlation between flights parameter
        :param adults: nb. of people on this ticket
        :param cabinclass: class of flight ticket
        :param simplify_compute: simplifies the computation in that it only simulates the last simulation date,
                                 options are: "take_last_only", "all_sim_dates"
        :param db_host: database host, where the market & flight data is located.
        """

        self.mkt_date = mkt_date
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
        self.db_host               = db_host

        super().__init__( mkt_date
                        , list(self.get_flights())
                        , K                = K
                        , rho              = rho
                        , simplify_compute = simplify_compute
                        , underlyer        = underlyer )

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
        """ Gets drift and vol for flights. It caches it, so that drift and vol are not re-fetching it.
        """

        dep_date, flight_id = flight_nb
        orig, dest, carrier = self._get_origin_dest_carrier_from_flight(flight_id)

        return get_drift_vol_from_db( dep_date
                                    , orig
                                    , dest
                                    , carrier
                                    , default_drift_vol = (500., 501.)  # TODO: FIX THIS PARAMETERS
                                    , db_host           = self.db_host )


class AirOptionMock(AirOptionSkyScanner):
    """ Air options computation for some mock flight data. The data are generated at random.
        IMPORTANT: JUST USED FOR TESTING.
    """

    def get_flights(self):
        """ Generates mock flight data - used for testing. Is a generator.
        """

        nb_flights = 15

        for flight_nb in range(1, nb_flights):
            yield ( np.random.random() * 100 + 100
                  , self.mkt_date + datetime.timedelta(days=flight_nb)
                  , 'UA' + str(flight_nb) )

    @functools.lru_cache(maxsize=128)
    def _drift_vol_for_flight(self, flight_nb: Tuple[datetime.date, str]) -> Tuple[float, float]:
        """ Fake drift/vol, just for testing.
        """

        return 100., 100.


class AirOptionFlightsExplicit(AirOptionFlights):
    """ Class that prices the AOTrade associated with list of flights.

    SOME GUIDANCE: use mkt_date: datetime.date(2016, 1, 1)
    """

    def __init__(self
                , mkt_date         : datetime.date
                , ao_flights       : List[Flight]
                , strike           : float
                , rho              : float = 0.95
                , simplify_compute : str   = 'take_last_only'
                , underlyer        : str   = 'n'):
        """ Computes the air option from the database.

        :param mkt_date: market date
        :param ao_flights: flights corresponding to the AOTrade.
        :param strike: strike for the AOTrade
        :param rho: correlation between flights parameter
        :param simplify_compute: simplifies the computation in that it only simulates the last simulation date,
                                 options are: "take_last_only", "all_sim_dates"
        :param underlyer: underlying model to use.
        """

        super().__init__( mkt_date
                        , [self.extract_prices(ao_flight) for ao_flight in ao_flights]
                        , K                = strike
                        , rho              = rho
                        , simplify_compute = simplify_compute
                        , underlyer        = underlyer )

    @staticmethod
    def extract_prices(ao_flight : Flight) -> Tuple[float, datetime.date, str]:
        """ Gets the prices and other data from the flight.

        :param ao_flight: flight information that you want info from.
        :returns: triple of price, datetime.date and flight id.
        """

        found_prices = ao_flight.prices  # prices found in the database
        if not found_prices:  # no prices found
            flight_price = 200.  # TODO: Some random price for now
        else:
            flight_price = found_prices[-1].price  # find the last price

        return flight_price, ao_flight.dep_date.date(), ao_flight.flight_id_long


class AirOptionsFlightsExplicitSky(AirOptionFlightsExplicit):
    """ Computes the air option from the data in the database, in particular
    flights between outbound_start and outbound_end, for the particular origin and destination.
    """

    def __init__( self
                , mkt_date  : datetime.date
                , origin    : str = 'SFO'
                , dest      : str = 'EWR'
                # next 4 - when do the (changed) flights occur
                , outbound_date_start : Optional[datetime.date] = None
                , outbound_date_end   : Optional[datetime.date] = None
                , inbound_date_start  : Optional[datetime.date] = None
                , inbound_date_end    : Optional[datetime.date] = None
                , K                   : float = 1600.
                , carrier             : str   = 'UA'
                , rho                 : float = 0.95
                , adults              : int   = 1
                , cabinclass          : str   = 'Economy'
                , simplify_compute    : str   = 'take_last_only'
                , underlyer           : str   = 'n'
                , return_flight       : bool  = False
                , recompute_ind       : bool  = False
                , correct_drift       : bool  = True
                , db_host             : str   = 'localhost' ):
        """
        :param origin: IATA code of the origin airport ('SFO')
        :param dest: IATA code of the destination airport ('EWR')
        :param outbound_date_start: start date for outbound flights to change to
        :param outbound_date_end: end date for outbound flights to change to
        :param inbound_date_start: start date for inbound flights to change to
        :param inbound_date_end: end date for inbound flights to change to
        :param K: option strike
        :param carrier: IATA code of the carrier
        :param rho: correlation between flights parameter
        :param adults: nb. of people on this ticket
        :param cabinclass: class of flight ticket
        :param simplify_compute: simplifies the computation in that it only simulates the last simulation date,
                                 options are: "take_last_only", "all_sim_dates"
        :param db_host: database host, where the market & flight data is located.
        """

        super().__init__( mkt_date
                        , self._get_flights(origin, dest, outbound_date_start, outbound_date_end, carrier)  # List[Flight]
                        , K
                        , rho
                        , simplify_compute = simplify_compute
                        , underlyer = underlyer)

    @staticmethod
    def _get_flights( origin              : str
                    , dest                : str
                    , outbound_date_start : datetime.date
                    , outbound_date_end   : datetime.date
                    , carrier             : str
                    , cabinclass          : str = 'economy'
                    , adults              : int = 1) -> List[Flight]:
        """ Get flights corresponding to the data that exist in the db.

        :param origin: IATA origin.
        :parma dest: IATA destination airport.
        :param outbound_date_start: start date for the flight search
        :param outbound_date_end: end date for the flight search
        :param carrier: IATA carrier id., like 'UA'
        :returns: list of all flights in the db corresponding to the parameters given.
        """

        session = create_session()

        return session.query(Flight)\
                      .filter_by(orig=origin, dest=dest, carrier=carrier)\
                      .filter(Flight.dep_date.between(outbound_date_start, outbound_date_end))\
                      .all()


class AirOptionFlightsFromDB(AirOptionFlightsExplicit):
    """ Class to fetch the trade from the database.

    SOME GUIDANCE: use mkt_date: datetime.date(2016, 1, 1)
    """

    def __init__(self
                , mkt_date         : datetime.date
                , ao_trade_id      : str
                , rho              : float = 0.95
                , simplify_compute : str   = 'take_last_only'
                , underlyer        : str   = 'n'
                , session         : Optional[str] = None):
        """ Computes the air option from the database.

        :param mkt_date: market date
        :param ao_trade_id: trade id for a particular AO trade we want.
        :param rho: correlation between flights parameter
        :param simplify_compute: simplifies the computation in that it only simulates the last simulation date,
                                 options are: "take_last_only", "all_sim_dates"
        :param underlyer: underlying model to use.
        :param session: session used for the fetching of trades database from where the AOTrade is fetched.
        """

        # database session
        session_used = create_session() if session is None else session

        ao_trade = session_used.query(AOTrade)\
                          .filter_by(position_id=ao_trade_id)\
                          .first()  # AOFlight object

        if ao_trade is None:
            raise AOTradeException(f'Trade number {ao_trade_id} could not be found.')

        super().__init__( mkt_date
                        , ao_trade.flights
                        , strike           = ao_trade.strike
                        , rho              = rho
                        , simplify_compute = simplify_compute
                        , underlyer        = underlyer )


def main():
    """ Example usage of some of the functions.
    """

    ao1 = AirOptionFlightsFromDB(datetime.date(2016, 1, 1), 1)
    print(ao1.PV())
