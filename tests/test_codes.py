# Test case for Air Option Estimate module
import unittest


from iata.codes import get_airline_code


class TestAoCodes(unittest.TestCase):

    def test_get_airline_code(self):
        """ Tests whether the airline codes are obtained correctly.
        """

        self.assertListEqual(get_airline_code('United'), ['UA'])
