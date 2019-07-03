#! /bin/bash

#curl localhost:5000/verify_airline?airline=United
#curl localhost:5000/verify_origin?origin=Newark
#curl localhost:5000/find_relevant_carriers?origin=EWR\&dest=SFO
#curl --header 'Content-Type: application/json' -d '{"nb_people": 1, "cabin_class": "Economy", "airline_name": "United", "option_start": "12/15/2016", "option_end": "12/16/2016", "origin": "SFO", "return_ow": "one_way", "ticket_price": 800, "outbound_start": "02/24/2018", "outbound_end": "02/25/2018", "dest": "EWR"}' localhost:5000/compute_option_now

curl --header 'Content-Type: application/json' -d '{"nb_people": 1, "cabin_class": "Economy", "airline_name": "United", "option_start": "12/15/2016", "option_end": "12/16/2016", "origin": "SFO", "return_ow": "one_way", "ticket_price": 800, "outbound_start": "02/24/2018", "outbound_end": "02/25/2018", "dest": "EWR"}' localhost:5000/compute_option
