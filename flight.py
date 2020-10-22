# flight class for ORM, trade access

from typing import Tuple

from sqlalchemy                 import Column, Integer, String, DateTime, ForeignKey, BigInteger, Table, Float, SmallInteger, Enum, create_engine
from sqlalchemy.orm             import relation, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from ao.ao_params import correct_drift_vol

AOORM = declarative_base()  # common base class


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

    @classmethod
    def from_skyscanner(cls, ss_entry):
        """ Constructs the object from skyscanner entry.
        """

        # TODO: correct here
        return cls( flight_id_long = ss_entry['flight_id_long']
                  , orig           = ss_entry['Itinerary']['origin']
                  , dest           = ss_entry['Itinerary']['destingation'] )

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
    strike                = Column(DateTime)
    carrier               = Column(String)
    nb_adults             = Column(Integer)
    cabinclass            = Column(String)

    flights = relation('Flight', secondary=t_trades_flights)


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

    engine = create_engine(db)
    session = sessionmaker(bind=engine)  # this is a class

    return session()

# examples:
# res1 = sess.query(Flight)
# tr1 = sess.query(AOTrade).filter_by(position_id=2).all()[0]
# tr2 = tr1.flights
# print(tr2)

# typical query which implements AOTrade:
# select fid.flight_id_long, fo.price
# from flight_ids fid, option_positions op, trades_flights tf, flights_ord fo
# where op.origin = 'SFO' and op.dest = 'EWR'
#      and op.position_id = tf.trade_id
#      and tf.flight_id = fid.flight_id
#      and fo.flight_id = tf.flight_id
