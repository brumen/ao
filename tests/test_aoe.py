import config
import numpy as np
import ao_estimate as aoe


fids = np.array(aoe.find_flight_ids()['flight_id'][0:100]).reshape((10,10))
aoe.array_buttons(fids)

