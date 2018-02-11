from ao_scripts.compute_option import compute_option
from werkzeug.datastructures import ImmutableMultiDict



form = ImmutableMultiDict([ ('nb_people', u'1')
                          , ('cabin_class', u'Economy')
                          , ('option_ret_start', u'12/25/2016')
                          , ('airline_name', u'UA')
                          , ('option_start', u'12/15/2016')
                          , ('origin_place', u'SFO')
                          , ('return_ow', u'return')
                          , ('option_ret_end', u'12/26/2016')
                          , ('ticket_price', u'800')
                          , ('outbound_end', u'02/25/2018')
                          , ('dest_place', u'EWR')
                          , ('outbound_start', u'02/24/2018')
                          , ('outbound_start_ret', u'03/04/2018')
                          , ('outbound_end_ret', u'03/05/2018')
                          , ('option_end', u'12/16/2016')])

print compute_option(form)