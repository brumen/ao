""" Flight class for ORM, trade access
"""

import datetime
import numpy as np

import pymysql
pymysql.install_as_MySQLdb()  # TODO: After a successful MySQLdb installation, both these lines should go

from typing import Tuple, List

from sqlalchemy                 import ( Column
                                       , Integer
                                       , String
                                       , DateTime
                                       , ForeignKey
                                       , BigInteger
                                       , Table
                                       , Float
                                       , SmallInteger
                                       , Enum
                                       , Date
                                       , Time
                                       , create_engine
                                       , )
from sqlalchemy.orm             import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base


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

    @staticmethod
    def _drift_vol( date_l  : List[datetime.date]
                  , price_l : List[float]
                  , model   : str   = 'n'
                  , dcf     : float = 385.25) -> Tuple[float, float, float]:
        """ Compute the drift and volatility of the normal/lognormal model.

        :param date_l:  list of dates
        :param price_l: list of prices at those dates
        :param model:   model selected: 'n' for normal, 'ln' for log-normal
        :param dcf: day-count factor.
        :returns: tuple of drift, vol, and average price.
        """

        date_diff = np.diff(np.array([(x - datetime.datetime.now()).seconds for x in date_l])) / (dcf * 86400)

        price_l_diff = np.diff(np.array(price_l))
        price_diff = price_l_diff / np.array(price_l[:-1]) if model == 'ln' else price_l_diff

        drift_over_sqdate = price_diff / np.sqrt(date_diff)

        drift_len = len(price_l)
        drift = np.sum(price_diff / date_diff) / drift_len

        vol_1 = np.sum(drift_over_sqdate ** 2)
        vol_2 = np.sum(drift_over_sqdate)
        vol = np.sqrt((vol_1 / drift_len - (vol_2 / drift_len) ** 2))

        avg_price = np.double(np.sum(price_l)) / drift_len

        return drift, vol, avg_price

    def drift_vol( self
                 , default_drift_vol : Tuple[float, float] = (500., 501.)
                 , fwd_value                               = None ) -> Tuple[float, float]:
        """ Pulls the drift and vol from database for the selected flight.

        :param default_drift_vol: correct the drift, if negative make it positive 500, or so.
        :param fwd_value: forward value used in case we want to correct the drift. If None, take the original drift.
        """

        if not self.prices:  # prices are empty
            return default_drift_vol

        # there are elements in self.prices
        dates   = [price_entry.as_of for price_entry in self.prices]
        prices_ = [price_entry.price for price_entry in self.prices]
        drift, vol, avg_price = self._drift_vol(dates, prices_)

        if drift <= 0:  # wrong drift
            return default_drift_vol

        return drift, vol


class FlightLive(AOORM):
    """ Live version of the Flights.
    """

    __tablename__ = 'flights_live'

    as_of       = Column(DateTime)  # designates when the flight was inserted.
    orig        = Column(String)
    dest        = Column(String)
    price       = Column(Float)
    flight_id   = Column(String)
    dep_date    = Column(Date)
    dep_time    = Column(Time)
    arr_date    = Column(DateTime)
    carrier     = Column(String)
    flight_nb   = Column(String)
    cabin_class = Column(Enum('economy', 'premium', 'business', 'first'))
    flights_primary_id = Column(BigInteger, primary_key=True)


# in between table that links Flight and AOTrade.
t_trades_flights = Table( 'trades_flights'
                        , AOORM.metadata
                        , Column('entry_id' , BigInteger, primary_key=True)
                        , Column('trade_id' , Integer   , ForeignKey('option_positions.position_id'))
                        , Column('flight_id', Integer   , ForeignKey('flight_ids.flight_id')        )
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
