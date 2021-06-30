# flight class for ORM, trade access

import datetime

import numpy as np

from typing import Tuple, Optional

from sqlalchemy                 import Column, Integer, String, DateTime, ForeignKey, BigInteger, Table, Float, SmallInteger, Enum, create_engine
from sqlalchemy.orm             import relation, sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base

from ao.ao_params import correct_drift_vol
from ao.air_option import AirOptionFlights
from ao.delta_dict import DeltaDict


AOORM = declarative_base()  # common base class


class Prices(AOORM):
    """ Description of the price for the flight.

    as_of: date for which this price entry was entered.
    reg_id: region id (for model computing purposes)
    flight_id: identifier of the flight in the db.
    prices_entry_id: primary key
    """

    __tablename__ = 'flights_ord'

    prices_entry_id = Column(BigInteger, primary_key=True)
    as_of           = Column(DateTime)
    price           = Column(Float)
    reg_id          = Column(Integer)
    flight_id       = Column(BigInteger, ForeignKey('flight_ids.flight_id'))


class Flight(AOORM):
    """ Description of the flight.

    flight_id: identifier of the flight in the db.
    flight_id_long: flight identifier from skyscanner.
    dep_date: departure date of the flight
    orig: IATA code of the origin airport ('EWR')
    dest: IATA code of the destination airport ('SFO')
    carrier: airline carrier, e.g. 'UA'
    """

    __tablename__ = 'flight_ids'

    flight_id      = Column(Integer, primary_key=True)
    flight_id_long = Column(String)
    orig           = Column(String)
    dest           = Column(String)
    dep_date       = Column(DateTime)
    arr_date       = Column(DateTime)
    carrier        = Column(String)

    prices = relationship('Prices')

    # TODO: REWRITE THE WHOLE THING HERE
    def drift_vol( self
                 , default_drift_vol : Tuple[float, float] = (500., 501.)
                 , fwd_value                               = None ) -> Tuple[float, float]:
        """ Pulls the drift and vol from database for the selected flight.

        :param default_drift_vol: correct the drift, if negative make it positive 500, or so.
        :param fwd_value: forward value used in case we want to correct the drift. If None, take the original drift.
        """

        # the same as the function get_drift_vol_from_db in ao_params.py
        session = create_session()
        drift_vol_params = session.query(AOParam).filter_by(orig=self.orig, dest=self.dest, carrier=self.carrier).all()

        if len(drift_vol_params) == 0:  # nothing in the list
            return default_drift_vol

        # at least one entry, check if there are many, select closest by as_of date
        # entry in form (datetime.datetime, drift, vol, avg_price)
        closest_date_params = sorted( drift_vol_params
                                    , key = lambda drift_vol_param: abs((self.dep_date - drift_vol_param.as_of).days))[0]

        return correct_drift_vol( closest_date_params.drift
                                , closest_date_params.vol
                                , default_drift_vol
                                , closest_date_params.avg_price
                                , fwd_value)


# in between table that links Flight and AOTrade.
t_trades_flights = Table( 'trades_flights'
                        , AOORM.metadata
                        , Column('entry_id' , BigInteger, primary_key=True)
                        , Column('trade_id' , Integer   , ForeignKey('option_positions.position_id'))
                        , Column('flight_id', Integer   , ForeignKey('flight_ids.flight_id')        )
                        , )


class AOTrade(AOORM):
    """ AirOptions trade.
    """

    __tablename__ = 'option_positions'

    position_id           = Column(Integer, primary_key=True)
    origin                = Column(String)
    dest                  = Column(String)
    option_start_date     = Column(DateTime)
    option_end_date       = Column(DateTime)
    option_ret_start_date = Column(DateTime)
    option_ret_end_date   = Column(DateTime)
    outbound_date_start   = Column(DateTime)
    outbound_date_end     = Column(DateTime)
    inbound_date_start    = Column(DateTime)
    inbound_date_end      = Column(DateTime)
    strike                = Column(Float)
    carrier               = Column(String)
    nb_adults             = Column(Integer)
    cabinclass            = Column(String)

    flights = relation('Flight', secondary=t_trades_flights)

    # cached value  TODO: perhaps a better way
    __aof = None

    def _aof(self, mkt_date : datetime.date) -> AirOptionFlights:
        """ AirOptionsFlight object extracted from it.

        :return: air option flights object from the parameters
        """

        if self.__aof:
            return self.__aof

        unpack_flights = []
        for flight in self.flights:
            price = flight.prices[-1].price
            fwd_date = flight.dep_date.date()
            flight_nb = flight.flight_id

            unpack_flights.append((price, fwd_date, flight_nb))

        self.__aof = AirOptionFlights(mkt_date, flights=unpack_flights, K=self.strike)
        return self.__aof

    def PV(self, mkt_date : datetime.date) -> float:
        """ Computes the present value of the AO trade.

        :param mkt_date: market date for the trade
        :returns: present value of the trade, for a particular market date.
        """

        return self._aof(mkt_date).PV( option_start_date     = self.option_start_date.date()
                                     , option_end_date       = self.option_end_date.date()
                                     , option_ret_start_date = self.option_ret_start_date.date()
                                     , option_ret_end_date   = self.option_ret_end_date.date()
                                     , )

    def PV01(self, mkt_date : datetime.date) -> DeltaDict[int, float]:
        """ Computes the PV01 of the trade for a particular market date.

        :param mkt_date: market date for which the delta is computed.
        :returns: delta of the trade with respect to individual flights in the trade.
        """

        return self._aof(mkt_date).PV01( option_start_date     = self.option_start_date.date()
                                       , option_end_date       = self.option_end_date.date()
                                       , option_ret_start_date = self.option_ret_start_date.date()
                                       , option_ret_end_date   = self.option_ret_end_date.date()
                                       , )


class AORegIds(AOORM):
    """ Region Id table reference.
    """

    __tablename__ = 'reg_ids'

    reg_id      = Column(Integer, primary_key=True)
    month       = Column(SmallInteger)
    tod         = Column(Enum('morning', 'afternoon', 'evening', 'night'))
    weekday_ind = Column(Enum('weekday', 'weekend'))


class AOParam(AOORM):
    """ Parameters for ao flights, volatility, drift, etc.
    """

    __tablename__ = 'params'

    param_id  = Column(BigInteger, primary_key=True)
    as_of     = Column(DateTime)
    orig      = Column(String(3))
    dest      = Column(String(3))
    carrier   = Column(String(2))
    drift     = Column(Float)
    vol       = Column(Float)
    avg_price = Column(Float)
    reg_id    = Column(BigInteger, ForeignKey('reg_ids.reg_id'))


def create_session(db : str = 'mysql://brumen@localhost/ao'):
    """ Creates a session to the database:
    # TODO: CHECK if this is a singleton pattern or not.

    :param db: database to create a connection to.
    """

    return sessionmaker(bind=create_engine(db))()


def select_random_flights( nb_flights : int = 10, db_session = None ):
    """ Selects random flights from the database

    """

    session = db_session if db_session is not None else create_session()
    rand_flights = set(np.random.randint(1, 1000, nb_flights))  # remove duplicates

    return session.query(Flight).filter(Flight.flight_id.in_(rand_flights)).all()  # all flights


def insert_random_flights(nb_positions : int = 10, nb_flights : Optional[int] = 10, strike : float = 200. ):
    """ Inserts number of positions in the database.

    :param nb_positions: number of positions to be inserted in the database.
    :param nb_flights: each position has this number of flights considered.
    :param db: database where positions are inserted.
    """

    session = create_session()

    # start_pos_id = session.query(AOTrade).count() + 1

    trades = [ AOTrade( flights     = select_random_flights(nb_flights=nb_flights, db_session=session)
                      , strike      = strike
                      , nb_adults   = 1
                      , cabinclass  = 'Economy' )
               for _ in range(nb_positions) ]

    for trade in trades:
        session.add(trade)
    session.commit()


# examples:
# res1 = sess.query(Flight)
# tr1 = sess.query(AOTrade).filter_by(position_id=2).all()[0]
# tr2 = tr1.flights
# print(tr2)

# deleting trades:
# sess = create_session()
# trade1 = sess.query(AOTrade).filter_by(position_id=8)[0]
# print(trade1.PV(datetime.date(2016, 1, 1)))
# print(trade1.PV01(datetime.date(2016, 1, 1)))

# sess.delete(trade1)
