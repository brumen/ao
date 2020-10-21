# flight class for ORM, trade access

from sqlalchemy                 import Column, Integer, String, DateTime, ForeignKey, BigInteger, Table, Float, SmallInteger, Enum
from sqlalchemy.orm             import relation
from sqlalchemy.ext.declarative import declarative_base

AOORM = declarative_base()  # common base class


class Flight(AOORM):
    """ Description of the flight.
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


class AODriftVol(AOORM):
    """ Class for drift & volatility.
    """

    __tablename__ = 'params'

    as_of     = Column(DateTime)
    orig      = Column(String)
    dest      = Column(String)
    carrier   = Column(String(2))  # TODO: CHECK IF THIS 2 MAKES SENSE
    drift     = Column(Float)  # TODO: CHeck if Float is right, in the table it's DOUBLE
    vol       = Column(Float)
    avg_price = Column(Float)
    reg_id    = Column(Integer)


class AORegIds(AOORM):
    """ Region Id table reference.
    """

    __tablename__ = 'reg_ids'

    reg_id = Column(Integer, primary_key=True)
    month  = Column(SmallInteger)
    tod    = Column(Enum)  # TODO: FINISH THIS ENUMS here and in the column below.
    weekday_ind = Column(Enum)


class AOParam(AOORM):
    """ Parameters for ao flights.
    """

    __tablename__ = 'params'

    param_id = Column(BigInteger, primary_key=True)
    as_of    = Column(DateTime)
    orig     = Column(String(3))
    dest     = Column(String(3))
    carrier  = Column(String(2))
    drift    = Column(Float)
    vol      = Column(Float)
    avg_price= Column(Float)
    reg_id   = Column(BigInteger, ForeignKey('reg_ids.reg_id'))


# example of using flights.
# from sqlalchemy     import create_engine
# from sqlalchemy.orm import sessionmaker
# engine = create_engine('mysql://brumen@localhost/ao')
# Session = sessionmaker(bind=engine)
# sess = Session()
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
