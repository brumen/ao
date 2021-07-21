""" IATA airline codes & IATA cities.
"""

from typing import List

from sqlalchemy                 import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

from ao.flight import create_session

AOORM = declarative_base()  # common base class


class IATACodes(AOORM):
    """ Iata codes for airline names.
    """

    __tablename__ = 'iata_codes'

    airline_name     = Column(String, primary_key=True)
    iata_code        = Column(String)
    three_digit_code = Column(Integer)
    icao             = Column(String)
    country          = Column(String)


class IATACity(AOORM):
    """ Iata codes for airport name
    """

    __tablename__ = 'iata_cities'

    city_code  = Column(String, primary_key=True)
    city_state = Column(String)
    city_name  = Column(String)


def get_airline_code( airline_name : str, session=None ) -> List[str]:
    """ Returns all the airline codes associated w/ airline

    :param airline_name: partial airline name, can be Adria, or ria.
    :param session: computer host where the database is.
    """

    session_used = session if session else create_session()

    return [x.iata_code
            for x in session_used.query(IATACodes).filter(IATACodes.airline_name.like(f'%{airline_name}%')).all()]


def get_airline_name( iata_code : str, session = None):

    session_used = session if session else create_session()

    return [x.airline_name
            for x in session_used.query(IATACodes).filter(IATACodes.iata_code.like(f'%{iata_code}%')).all()]


def get_city_code(city_name : str, session=None ):

    session_used = session if session else create_session()

    return [x.city_code
            for x in session_used.query(IATACity).filter(IATACity.city_name.like(f'%{city_name}%')).all()]


def get_city_name( city_code : str, session=None):

    session_used = session if session else create_session()

    return [x.city_name
            for x in session_used.query(IATACity).filter(IATACity.city_code.like(f'%{city_code}%')).all()]
