""" testing framework for Air option trades
"""

import datetime
import sys
from unittest import TestCase

if '/home/brumen/work/' not in sys.path:
    sys.path.append('/home/brumen/work/')

from ao.trade  import AirOptionFlights, create_session, AOTrade
from ao.flight import Flight


class TestTrade(TestCase):

    def test_positive_pv01(self):
        """ Tests if AirOptionFlights even runs, and tests some characteristics of option value
        """

        session = create_session()

        flights = session.query(Flight).filter(Flight.flight_id.in_([88, 114, 126, 333])).all()

        airof = AOTrade( flights = flights
                         , strike = 200.
                         , nb_adults = 1
                         , cabinclass = 'Economy')

        print(f"RES = {airof.PV(mkt_date=datetime.date(2016, 7, 1),nb_sim=500000)}")
        airof_pv01 = airof.PV01(mkt_date=datetime.date(2016, 7, 1), nb_sim = 500000)
        print(airof_pv01)
