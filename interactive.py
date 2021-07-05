""" Interactive part of the air options.
"""

from marisa_trie import Trie

from ao.iata.codes import IATACity, IATACodes
from ao.flight import create_session


def create_cities_trie(session = None):
    session_used = session if session is not None else create_session()

    return Trie([x.city_name for x in session_used.query(IATACity).all()])  # all cities


def create_iata_trie(session = None):
    session_used = session if session is not None else create_session()

    return Trie([x.airline_name for x in session_used.query(IATACodes).all()])  # all airlines


# querying using t.keys('a') - for all keys starting w/ 'a'
