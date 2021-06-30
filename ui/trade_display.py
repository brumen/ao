""" Display the trades and PV

"""
import datetime
import tkinter as tk
import numpy   as np
import pandas  as pd

from pandastable import Table
from typing      import List, Union

from ao.flight import AOTrade, create_session


# compute the pvs
def trade_pvs(mkt_date = datetime.date(2016, 1, 1), session = None) -> List[List[Union[int, float]]]:
    sess = create_session() if session is None else session
    all_trades = sess.query(AOTrade).all()

    all_trade_pv = []
    for trade in all_trades:
        trade_pv = trade.PV(mkt_date=mkt_date)
        trade_id = trade.position_id
        all_trade_pv.append([trade_id, trade_pv])

    return all_trade_pv

root = tk.Tk()
frame = tk.Frame(root)
frame.pack()
pt = Table(frame, showtoolbar=True, showstatusbar=True)

pt.model.df = pd.DataFrame(np.array(trade_pvs()))


pt.show()
root.mainloop()

