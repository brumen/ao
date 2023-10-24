""" Trade ORM
"""
import logging
import datetime
import numpy as np

from typing import Optional, Set, Dict, Tuple, Any, List

from sqlalchemy import Column, Integer, String, DateTime, Float
from sqlalchemy.orm import relation

from ao.air_option import AirOptionFlights
from ao.delta_dict import DeltaDict
from ao.flight import t_trades_flights, create_session, Flight, AOORM


logging.basicConfig(filename='/tmp/air_option.log')
logger = logging.getLogger(__name__)
logger.setLevel('INFO')


class AOTradeException(Exception):
    pass


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

    def _aof(self, mkt_date: datetime.date) -> AirOptionFlights:
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

    def aof_market(
            self,
            mkt_date: datetime.date,
            market: Dict[Tuple[str, datetime.date], float],
            default_params: Dict[str, Any],
    ) -> AirOptionFlights:
        """ Contructs the AirOptionFlights from the market ...

        """

        default_price = default_params['default_price']

        flights = []
        for flight in self.flights:
            # dep_date is datetime.datetime by default
            dep_date = flight.dep_date.date()
            flight_id = flight.flight_id
            carrier = flight.carrier
            flight_nb = f'{carrier}{flight_id}'
            logger.debug(f'Processing flight nb: {flight_nb}')

            if market is None:  # no idea about the market
                found_prices = flight.prices  # prices found in the database
                # find the last price, otherwise report a random price
                mkt_price = found_prices[-1].price if found_prices else default_price

            else:
                mkt_price = market.get((flight_nb, dep_date))
                if mkt_price is None:  # if market doesnt contain price
                    found_prices = flight.prices  # prices found in the database
                    # find the last price, otherwise report a random price
                    mkt_price = found_prices[-1].price if found_prices else default_price

            flights.append((mkt_price, dep_date, flight_nb))

        return AirOptionFlights(mkt_date, flights, self.strike)

    def _option_dates(self):
        osd = None if self.option_start_date is None else self.option_start_date.date()
        oed = None if self.option_end_date       is None else self.option_end_date.date()
        orsd = None if self.option_ret_start_date is None else self.option_ret_start_date.date()
        ored = None if self.option_ret_end_date   is None else self.option_ret_end_date.date()

        return osd, oed, orsd, ored

    def PV(self, mkt_date: datetime.date, nb_sim: int = 1000) -> float:
        """ Computes the present value of the AO trade.

        :param mkt_date: market date for the trade
        :param nb_sim: number of simulations to compute the PV.
        :returns: present value of the trade, for a particular market date.
        """

        osd, oed, orsd, ored = self._option_dates()

        return self._aof(mkt_date).PV(
            option_start_date=osd,
            option_end_date=oed,
            option_ret_start_date=orsd,
            option_ret_end_date=ored,
            nb_sim=nb_sim,
        )

    def PV01(self, mkt_date: datetime.date, nb_sim: int = 1000) -> DeltaDict:
        """ Computes the PV01 of the trade for a particular market date.

        :param mkt_date: market date for which the delta is computed.
        :returns DeltaDict[int, float]: delta of the trade with respect
           to individual flights in the trade.
        """

        osd, oed, orsd, ored = self._option_dates()

        return self._aof(mkt_date).PV01(
            option_start_date=osd,
            option_end_date=oed,
            option_ret_start_date=orsd,
            option_ret_end_date=ored,
            nb_sim=nb_sim,
        )


def select_random_flights( nb_flights : int = 10, db_session = None ):
    """ Selects random flights from the database

    """

    session = db_session if db_session is not None else create_session()
    rand_flights = set(np.random.randint(1, 1000, nb_flights).tolist())  # remove duplicates

    return session.query(Flight).filter(Flight.flight_id.in_(rand_flights)).all()  # all flights


def select_exact_flights( flights_to_add : Set, db_session = None ):
    """ Selects exact flights to add to the database.

    """

    session = db_session if db_session is not None else create_session()

    return session.query(Flight).filter(Flight.flight_id.in_(flights_to_add)).all()  # all flights


def insert_random_trades(
        nb_trades: int = 10,
        nb_flights_per_trade: Optional[int] = 10,
        strike: float = 200.,
):
    """ Inserts number of positions in the database.

    :param nb_trades: number of positions to be inserted in the database.
    :param nb_flights_per_trade: number of flights to be used
       for each trade position.
    :param strike: strike of the trade to be inserted.
    :returns: nothing, just inserts the number of trades
    """

    session = create_session()

    for _ in range(nb_trades):
        trade = AOTrade(
            flights=select_random_flights(
                nb_flights=nb_flights_per_trade,
                db_session=session,
            ),
            strike=strike,
            nb_adults=1,
            cabinclass='Economy',
        )
        session.add(trade)

    session.commit()


def insert_trade(
        flights_in_trade: Set,
        strike: float = 200.,
        session=None,
):
    """ Inserts number of positions in the database.

    :param flights_in_trade: flight_id of flights that you want to insert.
    :param strike: strike of the trade to be inserted.
    :returns: nothing, just inserts the number of trades
    """

    session = create_session() if session is None else session

    trade = AOTrade(
        flights=select_exact_flights(flights_in_trade, db_session=session),
        strike=strike,
        nb_adults=1,
        cabinclass='Economy',
    )

    session.add(trade)
    session.commit()


def delete_trades(trade_start_nb: int, trade_end_nb: int):
    """ Deletes trades in a range between trade_start_nb and trade_end_nb.
    """

    session = create_session()

    trades: List[AOTrade] = session\
        .query(AOTrade)\
        .filter(
            AOTrade.position_id.in_(
                range(trade_start_nb, trade_end_nb)
            )
        )\
        .all()

    for trade in trades:
        session.delete(trade)

    session.commit()


def main():
    """ Example usage of some of the functions.
    """

    t1 = create_session().query(AOTrade).filter_by(position_id=20)
    print(t1.PV(datetime.date(2016, 7, 1)))


# main()
