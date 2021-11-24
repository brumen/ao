#
# a widget for air option
#

import datetime
import tkinter as tk
import logging

import sys
sys.path.append('/home/brumen/work/')

from typing import Optional

from ao.air_option_derive     import AirOptionsFlightsExplicitSky as AOS
from ao.ui.autocomplete_combo import AutocompleteCombobox
from ao.iata.codes            import get_airline_code, get_city_code

# example:
#ao = AOS(datetime.date(2016, 9, 25), origin='SFO', dest='EWR', outbound_date_start=datetime.date(2016, 10, 1), outbound_date_end=datetime.date(2016, 10, 4), K = 100., carrier='UA')
# ao.PV()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# writing to sys.stdout
logger_handler = logging.StreamHandler(sys.stdout)
logger_handler.setLevel(logging.INFO)
logger.addHandler(logger_handler)


def entry_field(text_ : str, row : int, default_val : Optional[str] = None):
    """ Creates the tk.Entry component and a label component in a row row, with default value.

    :param text_: text to be displayed in the label.
    :param row: which row do the two elements belong.
    :param default_val: default value of the Entry field.
    :returns: Entry component of the two created.
    """

    tk.Label(top, text=text_).grid(row=row, column=0)

    if default_val is None:
        entry_comp = tk.Entry(top, bd=3)
    else:
        d_val = tk.StringVar()
        entry_comp = tk.Entry(top, bd=3, textvariable=d_val)
        d_val.set(default_val)

    entry_comp.grid(row=row, column=1)

    return entry_comp


def combo_field(text_, row, default_val = None, completion_list = None):
    """ Same as entry_comp, but w. AutocompleteCombobox component.
    """

    tk.Label(top, text=text_).grid(row=row, column=0)

    if default_val is None:
        combo_comp = AutocompleteCombobox(top)
    else:
        d_val = tk.StringVar()
        combo_comp = AutocompleteCombobox(top, textvariable=d_val)
        combo_comp.set_completion_list([] if completion_list is None else completion_list)
        d_val.set(default_val)

    combo_comp.grid(row=row, column=1)

    return combo_comp


# main file.
top = tk.Tk()

airports = get_city_code('')
airlines = get_airline_code('')
# origin
market_date    = entry_field('market date'   , 0, '2016-9-15')
origin         = combo_field('origin'        , 1, default_val='SFO', completion_list=airports)
dest           = combo_field('dest'          , 2, default_val='EWR', completion_list=airports)
strike         = entry_field('strike'        , 3, '100')
carrier        = combo_field('carrier'       , 4, default_val='UA', completion_list=airlines)
outbound_start = entry_field('outbound start', 5, '2016-10-1')
outbound_end   = entry_field('outbound end'  , 6, '2016-10-4')

# result field
tk.Label(top, text='result').grid(row=7, column=0)
res_var = tk.StringVar()
res = tk.Label(top, textvariable=res_var)
res_var.set(0)
res.grid(row=7, column=1)  # result field


compute_button = tk.Button(top
                           , text="compute"
                           , command=lambda: compute_ao(market_date.get(), origin.get(), dest.get(), strike.get(), carrier.get(), outbound_start.get(), outbound_end.get())
                           , ).grid(row=8, column=0, columnspan=2)


def compute_ao(market_date : str, origin : str, dest : str, strike : str, carrier : str, outbound_start : str, outbound_end : str):
    """ Handles the errors of the conversion, and output.
    """

    try:
        md = datetime.datetime.strptime(market_date, '%Y-%m-%d').date()
    except ValueError as ve:
        logger.error(f'Market date {market_date} could not be converted to date: {e}, {type(e)}')
        return None

    except Exception as e:
        logger.error(f'Market date {market_date} could not be converted to date: {e}, {type(e)}')
        return None

    try:
        os = datetime.datetime.strptime(outbound_start, '%Y-%m-%d').date()
    except Exception as e:
        logger.error(f'Outbound start date {outbound_start} could not be converted to date: {e}, {type(e)}')
        return None

    try:
        oe = datetime.datetime.strptime(outbound_end, '%Y-%m-%d').date()
    except Exception as e:
        logger.error(f'Outbound end date {outbound_end} could not be converted to date: {e}, {type(e)}')
        return None

    try:
        st = float(strike)
    except Exception as e:
        logger.error(f'Strike {strike} could not be converted to float: {e}, {type(e)}')
        return None

    aos = AOS( md
                , origin = origin
                , dest   = dest
                , outbound_date_start = os
                , outbound_date_end   = oe
                , K = st
                , carrier = carrier).PV()

    res_var.set(str(aos))


top.mainloop()
