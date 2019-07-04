def test_get_flight_data(self):
    """
    Tests whether the get_flight_data function even executes

    """

    res = ao.get_flight_data(outbound_date_start=self.outDate
                             , outbound_date_end=self.outDatePlusOne
                             , inbound_date_start=self.retDate
                             , inbound_date_end=self.retDatePlusOne)

    self.assertTrue(True)
