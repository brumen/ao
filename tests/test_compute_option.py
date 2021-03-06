# test file for compute option routine.

from unittest                import TestCase
from werkzeug.datastructures import ImmutableMultiDict

from ao.ao_scripts.compute_option import compute_option


class TestComputeOption(TestCase):
    """ Testing the final interface to compute option
    """

    sample_form = ImmutableMultiDict([ ('nb_people', u'1')
                          , ('cabin_class', u'Economy')
                          , ('option_ret_start', u'12/25/2016')
                          , ('airline_name', u'UA')
                          , ('option_start', u'12/15/2016')
                          , ('origin', u'SFO')
                          , ('return_ow', u'one_way')
                          , ('option_ret_end', u'12/26/2016')
                          , ('ticket_price', u'800')
                          , ('outbound_end', u'02/25/2018')
                          , ('dest', u'EWR')
                          , ('outbound_start', u'02/24/2018')
                          , ('outbound_start_ret', u'03/04/2018')
                          , ('outbound_end_ret', u'03/05/2018')
                          , ('option_end', u'12/16/2016')])

    def test_1(self):
        """ Checks if the function even runs.
        """

        res = compute_option(TestComputeOption.sample_form)  # a generator

        self.assertEqual(len(list(res)), 16)
