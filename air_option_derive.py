""" Basic AirOption computation class.
"""

import datetime
import numpy as np
import logging
import functools

from time                   import sleep
from typing                 import List, Tuple, Optional, Union, Dict, Generator
from sqlalchemy.orm.session import Session
from skyscanner.skyscanner  import Flights, FlightsCache

from ao.ao_params   import get_drift_vol_from_db
from ao.flight      import Flight, create_session, Prices, FlightLive
from ao.air_option  import AirOptionFlights, FLIGHT_TYPE
from ao.ds          import construct_date_range, convert_date_datedash
from ao.ao_codes    import COUNTRY, CURRENCY, LOCALE, skyscanner_api_key, livedb_delay


logging.basicConfig(filename='/tmp/air_option.log')
logger = logging.getLogger(__name__)
logger.setLevel('INFO')


class AirOptionSkyScanner(AirOptionFlights):
    """ Class for handling the air options from SkyScanner inputs.
    """

    WAIT_BTW_RETRIES = 5  # 5 seconds wait between retries of skyscanner calls.

    def __init__( self
                , mkt_date            : datetime.date
                , origin              : str = 'SFO'
                , dest                : str = 'EWR'
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
                , flights_include     : Optional[List] = None
                , correct_drift       : bool  = True
                , session             : Optional[Session] = None
                , curr_time           : Optional[datetime.datetime] = None ):
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
        :param underlyer: model underlyer, 'n' for normal, 'ln' for log-normal.
        :param return_flight: indicator whether return flight is considered.
        :param flights_include: list of flights to include, if None, include all flights.
        :param correct_drift: indicator whether to correct the drift.
        :param session: session used to interact w/ the database.
        :param curr_time: time selected for the current market date, used for fetching data from live data.
        """

        self.mkt_date = mkt_date
        self.origin = origin
        self.dest   = dest
        self.outbound_date_start = outbound_date_start
        self.outbound_date_end   = outbound_date_end
        self.inbound_date_start  = inbound_date_start
        self.inbound_date_end    = inbound_date_end
        self.carrier             = carrier
        self.cabinclass          = cabinclass
        self.adults              = adults
        self.return_flight       = return_flight
        self._correct_drift       = correct_drift
        self._flights_include      = flights_include
        self._session              = session if session else create_session()

        self._curr_time = curr_time if curr_time else datetime.datetime.fromisoformat(self.mkt_date.isoformat())

        super().__init__( mkt_date
                        , None  # flights are initialized w. None, lazy evaluation of lists.
                        , K                = K
                        , rho              = rho
                        , simplify_compute = simplify_compute
                        , underlyer        = underlyer )

    @property
    def flights(self):
        if self._flights:
            return self._flights

        # get flights and return them
        flights = self.get_flight_data(origin                = self.origin
                                       , dest                = self.dest
                                       , outbound_date_start = self.outbound_date_start
                                       , outbound_date_end   = self.outbound_date_end
                                       , inbound_date_start  = self.inbound_date_start
                                       , inbound_date_end    = self.inbound_date_end
                                       , carrier             = self.carrier
                                       , cabinclass          = self.cabinclass
                                       , adults              = self.adults
                                       , return_flight       = self.return_flight
                                       , correct_drift       = self._correct_drift
                                       , curr_time           = self._curr_time
                                       , )

        self._flights = list(flights) if not self.return_flight else (list(flights[0]), list(flights[1]))
        return self._flights

    @classmethod
    def get_flight_data ( cls
                        , flights_include     : Optional[List] = None
                        , origin              : str = 'SFO'
                        , dest                : str = 'EWR'
                        , outbound_date_start : Optional[datetime.date] = None
                        , outbound_date_end   : Optional[datetime.date] = None
                        , inbound_date_start  : Optional[datetime.date] = None
                        , inbound_date_end    : Optional[datetime.date] = None
                        , carrier             : str = 'UA'
                        , cabinclass          : str = 'Economy'
                        , adults              : int = 1
                        , return_flight       : bool = False
                        , correct_drift       : bool = True
                        , insert_into_livedb  : bool = True
                        , curr_time           : Optional[datetime.datetime] = None ) -> Union[ List[FLIGHT_TYPE], Tuple[List[FLIGHT_TYPE], List[FLIGHT_TYPE]]]:
        """ Get flight data for the parameters specified

        :param flights_include:      if None - include all flights
                                     if specified, then only consider the flights in flights_include
        :param origin: IATA code of the origin airport, e.g.  'SFO'
        :param dest: IATA code of the destination airport, e.g.  'SFO'
        :param outbound_date_start: start date of the outbound flights
        :param outbound_date_end: end date of the outbound flights
        :param inbound_date_start: start date of the inbound (return) flights
        :param inbound_date_end: end date of the inbound (return) flights
        :param carrier: IATA code of the carrier, e.g. 'UA'
        :param cabinclass: cabin class, like 'Economy'
        :param adults: number of adults for the flight
        :param return_flight: indicator for return flight
        :param curr_time: current time.
        """

        # departure flights.
        dep_flights = cls._obtain_flights( origin
                                          , dest
                                          , carrier
                                          , construct_date_range(outbound_date_start, outbound_date_end)
                                          , flights_include if (not return_flight) else (None if return_flight is None else flights_include[0])
                                          , cabinclass         = cabinclass
                                          , adults             = adults
                                          , insert_into_livedb = insert_into_livedb
                                          , correct_drift      = correct_drift
                                          , curr_time          = curr_time)

        dep_flights_used = [(flight.price, flight.dep_date, f'{flight.carrier}{flight.flight_nb}')
                            for flight in dep_flights]

        if not return_flight:  # departure flights, always establish
            return dep_flights_used

        # return flight
        ret_flights = cls._obtain_flights( dest
                                          , origin
                                          , carrier
                                          , construct_date_range(inbound_date_start, inbound_date_end)
                                          , None if not flights_include else flights_include[1]
                                          , cabinclass         = cabinclass
                                          , adults             = adults
                                          , insert_into_livedb = insert_into_livedb
                                          , correct_drift      = correct_drift
                                          , curr_time          = curr_time)

        ret_flights_used = [(flight.price, flight.dep_date, f'{flight.carrier}{flight.flight_nb}')
                            for flight in ret_flights]

        return dep_flights_used, ret_flights_used

    @classmethod
    def _find_carrier(cls, carriers: List[str], carrier_id: str) -> Optional[str]:
        """ Finds the carrier from the ID list

        :param carriers: list of carriers
        :param carrier_id: carrier one is searching for
        :returns: Code of the carrier info if found, else None
        """

        for carrier_info in carriers:
            if carrier_id == carrier_info['Id']:
                return carrier_info['Code']

        return None  # None indicates failure

    @classmethod
    def get_itins ( cls
                  , origin: str
                  , dest: str
                  , outbound_date: datetime.date
                  , includecarriers: Union[List[str], None] = None
                  , cabinclass: str = 'Economy'
                  , adults: int = 1
                  , use_cache: bool = False
                  , nb_tries: int = 1
                  , max_nb_tries: int = 5) -> Union[Dict, None]:
        """ Returns itineraries for the selection from the Skyscanner API.

        :param origin: IATA code of the flight origin airport (e.g. 'SIN', or 'SFO')
        :param dest: IATA code of the flight destination airport (e.g. 'KUL', or 'EWR')
        :param outbound_date: date for flights to fetch
        :param includecarriers: IATA code of the airlines to use, if None, all airlines
        :param cabinclass: one of the following: Economy*, PremiumEconomy, Business, First
        :param adults: number of adults to get
        :param use_cache: whether to use Skyscanner cache for ticket pricer. This is not the local db, just cache part of Skyscanner
        :param nb_tries: number of tries that one tries to get a connection to SkyScanner
        :param max_nb_tries: max number of tries that it attempts.
        :returns:             Resulting flights from SkyScanner, dictionary structure:
                              'Itineraries'
                              'Currencies'
                              'Agents'
                              'Carriers'
                              'Query'
                              'Segments'
                              'Places'
                              'SessionKey'
                              'Legs'
                              'Status'
                   if no flights could be found, return None
        """

        params_all = dict(country=COUNTRY
                          , currency=CURRENCY
                          , locale=LOCALE
                          , originplace=f'{origin}-sky'
                          , destinationplace=f'{dest}-sky'
                          , outbounddate=convert_date_datedash(outbound_date)
                          , cabinclass=cabinclass
                          , adults=adults
                          , stops=0)  # only direct flights

        if includecarriers is not None:
            params_all['includecarriers'] = includecarriers

        if not use_cache:
            flights_service = Flights(skyscanner_api_key)
            query_fct = flights_service.get_result
        else:
            flights_service = FlightsCache(skyscanner_api_key)
            query_fct = flights_service.get_cheapest_price_by_route
            # query_fct = flights_service.get_cheapest_quotes
            # query_fct = flights_service.get_cheapest_price_by_date
            # query_fct = flights_service.get_grid_prices_by_date
            params_all['market'] = COUNTRY  # add this field

        try:
            return query_fct(**params_all).parsed

        except (ConnectionError, AttributeError):
            sleep(cls.WAIT_BTW_RETRIES)  # wait 5 secs
            if nb_tries <= max_nb_tries:
                return cls.get_itins(origin=origin
                                 , dest=dest
                                 , outbound_date=outbound_date
                                 , includecarriers=includecarriers
                                 , cabinclass=cabinclass
                                 , adults=adults
                                 , nb_tries=nb_tries + 1
                                 , max_nb_tries=max_nb_tries)

            return None  # this is handled appropriately in the get_ticket_prices

    @classmethod
    def get_ticket_prices ( cls
                          , origin        : str
                          , dest          : str
                          , outbound_date : datetime.date
                          , include_carriers =None
                          , cabinclass    : str = 'Economy'
                          , adults        : int = 1
                          , use_cache     : bool = False
                          , insert_into_livedb: bool = False
                          , session: Optional[Session] = None
                          , local_only : bool = True
                          , curr_time : Optional[datetime.datetime] = None
                          , ) -> Union[None, List[FlightLive]]:
        """ Returns the list of live flights.

        :param origin: IATA code of the origin airport 'SIN'
        :param dest: IATA code of the destination airport 'KUL'
        :param outbound_date: outbound date # TODO: remove: in dash format '2017-02-15'
        :param include_carriers: IATA code of a _SINGLE_ airline code
        :param cabinclass: cabin class of the flight ticket (one of 'Economy', 'Business')
        :param adults: number of adult tickets booked
        :param: use_cache: bool indicator to signal to SkyScanner api to use cache.
        :param insert_into_livedb: indicator whether to insert the fetched flight into the livedb
        :param session: mysqlalchemy session, if None, one is made up directly in the function.
        :param local_only: use only prices stored in the local db, dont use Skyscanner.
        :returns: None if no results, otherwise a list of FlightLive objects
        """

        session_used = session if session else create_session()

        flights_in_ldb = session_used.query(FlightLive) \
            .filter_by(orig=origin
                       , dest=dest
                       , dep_date=outbound_date
                       , cabin_class=cabinclass) \
            .filter(FlightLive.as_of > (curr_time - livedb_delay).isoformat())

        if include_carriers is not None:
            flights_in_ldb = flights_in_ldb.filter_by(carrier=include_carriers)

        flights_in_ldb = flights_in_ldb.all()

        if flights_in_ldb or local_only:  # we have this in the database
            return flights_in_ldb

        # If flights not found in the cached db, continue with skyscanner fetch
        result = cls.get_itins( origin          = origin
                              , dest            = dest
                              , outbound_date   = outbound_date
                              , includecarriers = include_carriers
                              , cabinclass      = cabinclass
                              , adults          = adults
                              , use_cache       = use_cache
                              , )

        if not result:
            return None

        return cls._flights_skyscanner(result)  # TODO: IMPROVE THIS LIST

    @classmethod
    def _flights_skyscanner(cls, result) -> Generator[FlightLive, None, None]:
        """ Obtain prices from skyscanner results.

        :param result: Skyscanner results, in the following form:
        {
  "SessionKey": "ab5b948d616e41fb954a4a2f6b8dde1a_ecilpojl_7CAAD17D0CFC34BFDE68DEBFDFD548C7",
  "Query": {
    "Country": "GB",
    "Currency": "GBP",
    "Locale": "en-gb",
    "Adults": 1,
    "Children": 0,
    "Infants": 0,
    "OriginPlace": "2343",
    "DestinationPlace": "13554",
    "OutboundDate": "2017-05-30",
    "InboundDate": "2017-06-02",
    "LocationSchema": "Default",
    "CabinClass": "Economy",
    "GroupPricing": false
  },
  "Status": "UpdatesComplete",
  "Itineraries": [
    {
      "OutboundLegId": "11235-1705301925--32480-0-13554-1705302055",
      "InboundLegId": "13554-1706020700--32480-0-11235-1706020820",
      "PricingOptions": [
        {
          "Agents": [
            4499211
          ],
          "QuoteAgeInMinutes": 0,
          "Price": 83.41,
          "DeeplinkUrl": "http://partners.api.skyscanner.net/apiservices/deeplink/v2?_cje=jzj5DawL5zJyT%2bnfe1..."
        },
        ...
        ],
      "BookingDetailsLink": {
        "Uri": "/apiservices/pricing/v1.0/ab5b948d616e41fb954a4a2f6b8dde1a_ecilpojl_7CAAD17D0CFC34BFDE68DEBFDFD548C7/booking",
        "Body": "OutboundLegId=11235-1705301925--32480-0-13554-1705302055&InboundLegId=13554-1706020700--32480-0-11235-1706020820",
        "Method": "PUT"
      }
    },
    ...
   ],
  "Legs": [
    {
      "Id": "11235-1705300650--32302,-32480-1-13554-1705301100",
      "SegmentIds": [
        0,
        1
      ],
      "OriginStation": 11235,
      "DestinationStation": 13554,
      "Departure": "2017-05-30T06:50:00",
      "Arrival": "2017-05-30T11:00:00",
      "Duration": 250,
      "JourneyMode": "Flight",
      "Stops": [
        13880
      ],
      "Carriers": [
        885,
        881
      ],
      "OperatingCarriers": [
        885,
        881
      ],
      "Directionality": "Outbound",
      "FlightNumbers": [
        {
          "FlightNumber": "290",
          "CarrierId": 885
        },
        {
          "FlightNumber": "1389",
          "CarrierId": 881
        }
      ]
    },
    ...
   ],
   "Segments": [
    {
      "Id": 0,
      "OriginStation": 11235,
      "DestinationStation": 13880,
      "DepartureDateTime": "2017-05-30T06:50:00",
      "ArrivalDateTime": "2017-05-30T07:55:00",
      "Carrier": 885,
      "OperatingCarrier": 885,
      "Duration": 65,
      "FlightNumber": "290",
      "JourneyMode": "Flight",
      "Directionality": "Outbound"
    },
    ...
  ],
    "Carriers": [
    {
      "Id": 885,
      "Code": "BE",
      "Name": "Flybe",
      "ImageUrl": "http://s1.apideeplink.com/images/airlines/BE.png",
      "DisplayCode": "BE"
    },
    ...
  ],
  "Agents": [
    {
      "Id": 1963108,
      "Name": "Mytrip",
      "ImageUrl": "http://s1.apideeplink.com/images/websites/at24.png",
      "Status": "UpdatesComplete",
      "OptimisedForMobile": true,
      "BookingNumber": "+448447747881",
      "Type": "TravelAgent"
    },
    ...
  ],
  "Places": [
    {
      "Id": 11235,
      "ParentId": 2343,
      "Code": "EDI",
      "Type": "Airport",
      "Name": "Edinburgh"
    },
    ...
  ],
  "Currencies": [
    {
      "Code": "GBP",
      "Symbol": "£",
      "ThousandsSeparator": ",",
      "DecimalSeparator": ".",
      "SymbolOnLeft": true,
      "SpaceBetweenAmountAndSymbol": false,
      "RoundingCoefficient": 0,
      "DecimalDigits": 2
    },
    ...
  ]
}
        :returns: generator where each item is a FlightLive for the results.
        """

        time_now = datetime.datetime.now()

        for itinerary, leg in zip(result['Itineraries'], result['Legs']):

            flight_num_all = leg['FlightNumbers']

            if len(flight_num_all) == 1:  # indicator if the flight is direct, the other test case is missing
                carrier = cls._find_carrier(result['Carriers'], leg['Carriers'][0])  # carriers = all carriers, leg['carriers'] are id of carrier
                price = itinerary['PricingOptions'][0]['Price']  # TODO: THIS PRICE CAN BE DIFFERENT
                flight_num = flight_num_all[0]['FlightNumber']  # TODO: HOW DO WE KNOW THAT WE HAVE THIS??

                # leg['Departure'] is departure date
                # flights.append((leg['Id'], leg['Departure'], leg['Arrival'], price, carrier + flight_num))

                # prices - TODO: FIX THIS
                price = Prices( as_of  = time_now
                              , price  = price
                              , reg_id    = flight_num
                              , flight_id = carrier + flight_num)

                yield FlightLive( as_of     = time_now
                                , flight_id = leg['Id']
                                , prices    = price
                                , dep_date  = leg['Departure']  # leg['Departure'] is departure date
                                , )

    @classmethod
    def _obtain_flights ( cls
                       , origin            : str
                       , dest              : str
                       , carrier           : str
                       , date_range        : List[datetime.date]
                       , flights_include   : List
                       , cabinclass        : str = 'Economy'
                       , adults            : int = 1
                       , insert_into_livedb : bool = True
                       , correct_drift      : bool = True
                       , curr_time          : Optional[datetime.datetime] = None
                       , ) -> List[Union[Flight, FlightLive]]:
        """ Get the flights for outbound and/or inbound flight.

        :param origin: origin airport of flights, IATA code (like 'EWR')
        :param dest: dest airport of flights, IATA code (like 'SFO')
        :param carrier: IATA code of the carrier considered
        :param date_range:   input/output date range _minus (with - sign)
                              output of function construct_date_range(outbound_date_start, outbound_date_end)
        :param correct_drift: whether to correct the drift, as described in the documentation
        :param cabinclass: cabin class, one of 'Economy', ...
        :param adults: number of adults
        :param insert_into_livedb: whether to insert the obtained flights into livedb
        :returns: a tuple of:
                  F_v : vector of ticket prices
                  s_v : vector of ticket vols
                  d_v : vector of ticket drifts
                  flights_dep : vector of ticket dates (maturities of these forwards)
        """

        all_flights = []
        for out_date in date_range:
            all_flights.extend(cls.get_ticket_prices( origin              = origin
                                                    , dest               = dest
                                                      , outbound_date      = out_date
                                                      , include_carriers   = carrier
                                                      , cabinclass         = cabinclass
                                                      , adults             = adults
                                                      , insert_into_livedb = insert_into_livedb
                                                      , curr_time          = curr_time
                                                      , ) )

        return all_flights

    @functools.lru_cache(maxsize=128)
    def _drift_vol_for_flight(self, flight_nb: Tuple[datetime.date, str]) -> Tuple[float, float]:
        """ Gets drift and vol for flights. It caches it, so that drift and vol are not re-fetching it.

        :param flight_nb:
        :returns: drift and volatility for the selected flight.
        """

        dep_date, flight_id = flight_nb
        orig, dest, carrier = self._get_origin_dest_carrier_from_flight(flight_id)

        return get_drift_vol_from_db( dep_date
                                    , orig
                                    , dest
                                    , carrier
                                    , default_drift_vol = (500., 501.)  # TODO: FIX THIS PARAMETERS
                                    , session = self._session )


class AirOptionMock(AirOptionSkyScanner):
    """ Air options computation for some mock flight data. The data are generated at random.
        IMPORTANT: JUST USED FOR TESTING.
    """

    # TODO: this method should be class method, but it is regular for convenience - this is
    #       used for testing only.
    def get_flight_data ( self
                        , flights_include     : Optional[List] = None
                        , origin              : str = 'SFO'
                        , dest                : str = 'EWR'
                        , outbound_date_start : Optional[datetime.date] = None
                        , outbound_date_end   : Optional[datetime.date] = None
                        , inbound_date_start  : Optional[datetime.date] = None
                        , inbound_date_end    : Optional[datetime.date] = None
                        , carrier             : str = 'UA'
                        , cabinclass          : str = 'Economy'
                        , adults              : int = 1
                        , return_flight       : bool = False
                        , correct_drift       : bool = True
                        , insert_into_livedb  : bool = True
                        , curr_time           : Optional[datetime.datetime] = None ) -> Union[ List[FLIGHT_TYPE], Tuple[List[FLIGHT_TYPE], List[FLIGHT_TYPE]]]:
        """ Generates mock flight data - used for testing. Is a generator.
        """

        nb_flights = 15  # fictional number of flights for testing purposes.

        dep_flights = [ ( np.random.random() * 100 + 100  # random price, for fun.
                        , self.mkt_date + datetime.timedelta(days=flight_nb)
                        , f'UA{str(flight_nb)}' )
                        for flight_nb in range(1, nb_flights) ]

        if not return_flight:
            return dep_flights

        ret_flights = [ ( np.random.random() * 100 + 100  # random price, for fun.
                        , self.mkt_date + datetime.timedelta(days=flight_nb)
                        , f'UA{str(flight_nb)}' )
                        for flight_nb in range(nb_flights, 2*nb_flights-1) ]

        return dep_flights, ret_flights

    @functools.lru_cache(maxsize=128)
    def _drift_vol_for_flight(self, flight_nb: Tuple[datetime.date, str]) -> Tuple[float, float]:
        """ Fake drift/vol, just for testing.
        """

        return 100., 100.


class AirOptionsFlightsExplicitSky(AirOptionSkyScanner):
    """ Computes the air option from the data in the database, in particular
        flights between outbound_start and outbound_end, for the particular origin and destination.
    """

    @classmethod
    def get_flight_data ( cls
                        , flights_include     : Optional[List] = None
                        , origin              : str = 'SFO'
                        , dest                : str = 'EWR'
                        , outbound_date_start : Optional[datetime.date] = None
                        , outbound_date_end   : Optional[datetime.date] = None
                        , inbound_date_start  : Optional[datetime.date] = None
                        , inbound_date_end    : Optional[datetime.date] = None
                        , carrier             : str = 'UA'
                        , cabinclass          : str = 'Economy'
                        , adults              : int = 1
                        , return_flight       : bool = False
                        , correct_drift       : bool = True
                        , insert_into_livedb  : bool = True
                        , curr_time           : Optional[datetime.datetime] = None ) -> Union[ List[FLIGHT_TYPE], Tuple[List[FLIGHT_TYPE], List[FLIGHT_TYPE]]]:
        """ Get flights corresponding to the data that exist in the db.

        :param origin: IATA origin.
        :parma dest: IATA destination airport.
        :param outbound_date_start: start date for the flight search
        :param outbound_date_end: end date for the flight search
        :param carrier: IATA carrier id., like 'UA'
        :returns: list of all flights in the db corresponding to the parameters given.
        """

        session = create_session()

        flights =  session.query(Flight)\
                      .filter_by(orig=origin, dest=dest, carrier=carrier)\
                      .filter(Flight.dep_date.between(outbound_date_start, outbound_date_end))\
                      .all()

        for flight in flights:
            flight_id = flight.flight_id

            prices = flight.prices
            if not prices:  # list is empty
                raise RuntimeError(f'Cant find price for flight {flight_id}.')

            yield prices[-1].price, flight.dep_date.date(), flight.flight_id


# def _reorganize_ticket_prices(cls, flights: List[FlightLive]) -> Dict[datetime.date, Dict[str, Dict[str, FlightLive]]]:
#     """ Reorganize the flights by levels:
#        day (datetime.date)
#          time of day (morning, afternoon)
#             hour (datetime.time)
#
#         insert ((date, hour), (arr_date, arr_hour), price, flight_id) into dict d
#
#     :param flights: Itinerary in the form of a list of ((u'2016-10-28', u'19:15:00'), 532.),
#                    where the first is the departure date, second departure time, third flight price
#     :returns: multiple level dictionary
#     """
#
#     # get the days from the list of
#     # TODO: CHECK ITINERARIES IN THE UPSTREAM FUNCTION
#     # dep_day_hour = [(x[1].split('T'), x[2].split('T'), x[3], x[0], x[4]) for x in itin]
#
#     reorg_tickets = dict()
#
#     # for (date_dt, hour), (arr_date, arr_hour), price, flight_id, flight_num in dep_day_hour:
#     for flight in flights:
#         hour = flight.as_of.time()
#         time_of_day = get_tod(hour)
#         date_ = flight.as_of.date()
#
#         # part to insert into dict d
#         if date_ not in reorg_tickets.keys():
#             reorg_tickets[date_] = dict()
#
#         if time_of_day not in reorg_tickets[date_].keys():
#             reorg_tickets[date_][time_of_day] = dict()
#
#         # TODO: MISSING STUFF
#         price = None
#         arr_hour = None
#         reorg_tickets[date_][time_of_day][hour] = (
#         flight.flight_id, date_, hour, flight.arr_date, arr_hour, price, flight.flight_id, True)
#
#     return reorg_tickets
