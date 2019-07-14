# display flight module
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tkinter     import Frame, Button, Label
from scipy.stats import norm  # quantiles of normal distr. norm.ppf(x)

from ao_codes            import LARGE_DRIFT, DCF
from mysql_connector_env import MysqlConnectorEnv


logger = logging.getLogger(__name__)


def compute_conf_band( x0
                     , ts
                     , drift
                     , vol
                     , quant = 0.9):
    """Constructs mean, confidence bands for log-normal process starting from x0,

    :param ts: time-series
    :param x0: starting value of time series
    :param drift, vol: drift, vol of the process
    :param quant: quantile to which to display bounds
    :returns:
    """

    mean_ts = x0 + ts * drift
    quant = norm.ppf(0.5 - quant/2.)
    deviat_ts_pos = vol * np.sqrt(ts) * quant
    lower_q_ts = mean_ts + deviat_ts_pos
    upper_q_ts = mean_ts - deviat_ts_pos  # upper and lower are symmetric

    return mean_ts, lower_q_ts, upper_q_ts


def plot_flight_prices( df1
                        , drift = 100.
                        , vol   = 200. ):
    """
    uses df1 from flight_price_get to plot flight prices

    """
    # compute date diffs
    df1d = df1['as_of'].diff()  # this will express differences in timedelta

    # construct time series in normalized units
    ts = np.empty(len(df1d))
    ts[0] = 0.
    ts[1:] = np.array([x.total_seconds() / (86400 * DCF)
                       for x in df1d[1:]])  # numerical value of days
    ts = ts.cumsum()

    mean_ts, lower_q_ts, upper_q_ts = compute_conf_band(df1['price'][0], ts, drift, vol)

    # print ts, type(ts), fitted_prices, df1['price']
    data_df = np.array([np.array(df1['price']), mean_ts, lower_q_ts, upper_q_ts]).transpose()
    legends = ['price', 'mean', 'q05', 'q95']
    df2 = pd.DataFrame(data=data_df, index=ts, columns=legends)
    ax1 = df2.plot()
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines[:4], legends, loc='best')
    plt.show(block=False)

    return df2, ax1


def flight_price_get(flight_id : int):
    """
    Returns prices from flight number id and computes drift and vol for
    those prices

    :param flight_id: Flight id to display prices in the database for
    :returns:         tuple of data_frame, (drift, vol) computed for the flight_id
    :rtype:           tuple of DataFrame, (double, double)
    """

    flights_prices_str = """
    SELECT * 
    FROM flights_ord 
    WHERE flight_id = {0}
    ORDER BY as_of
    """.format(flight_id)

    orig_dest_str = """
    SELECT orig, dest, carrier 
    FROM flight_ids 
    WHERE flight_id = {0}
    """.format(flight_id)

    with MysqlConnectorEnv() as m_conn:

        df1 = pd.read_sql_query( flights_prices_str
                               , m_conn
                               , parse_dates = {'as_of': '%Y-%m-%d'})
        df2 = pd.read_sql_query( orig_dest_str
                               , m_conn)  # this is unique

        reg_id = df1['reg_id'][0]  # they are all the same, take first

        orig, dest, carrier = df2['orig'][0], df2['dest'][0], df2['carrier'][0]

        df3 = pd.read_sql_query( "SELECT drift, vol FROM params WHERE orig = '{0}' AND dest = '{1}' AND carrier = '{2}' AND reg_id = '{3}'".format(orig, dest, carrier, reg_id)
                               , m_conn)  # this is unique

        drift, vol = df3['drift'][0], df3['vol'][0]

    return df1, (drift, vol)


def plot_from_flight_id(flight_id):
    """
    Plots the graph of actual, mean, quantile prices from flight_id

    :param flight_id: id of the flight plotted
    :type flight_id:  int
    :returns:         plots the flight
    :rtype:           None
    """

    df1, drift_vol = flight_price_get(flight_id)

    if len(df1) > 1:
        drift, vol = drift_vol
        df2,   ax1 = plot_flight_prices(df1, drift=drift)
        return df1, df2, ax1
    else:
        print ("No flights registered for flight_id {0}".format(flight_id))


class ArrayButtons(Frame):

    def __init__( self
                  , mat
                  , master=None):
        """

        :params mat: matrix of rows, columns of flight numbers, like 1533531
        """
        Frame.__init__(self, master)
        self.pack()
        self.mat = mat   # holding the matrix

        self.btn = [[0 for y in range(len(mat[x]))] for x in range(len(mat))]

        for x in range(len(mat)):
            for y in range(len(mat[x])):
                self.btn[x][y] = Button( self
                                         , text    = mat[x][y]
                                         , command = lambda x1=x, y1=y: self.display_graph(x, y))
                self.btn[x][y].grid(column=x, row=y)

    def display_graph(self, x1, y1):
        """
        plots the graph TODO: FILL IN HERE

        :param x1: x-index of the matrix to display
        :type x1:  int
        :param y1: y-index of the matrix to display
        :type y1:  int
        """
        plot_from_flight_id(flight_id=self.mat[x1][y1])


class ArrayTextButtons(Frame):
    """
    mat ... matrix of rows, columns
    if there are more than 30 items, rearrange them in columns of 30
    """

    def __init__(self, text_fid_mat, master=None):
        """
        display flights w/ all the goodies

        :param f_l: flight list in the form [(some_text, flight_nb), ...]
                      some text can be anything you construct
        """

        Frame.__init__(self, master)
        self.pack()
        self.text_fid_mat = text_fid_mat   # list of (text, fid)
        self.nb_flights = len(text_fid_mat)
        self.nb_columns = self.nb_flights /30 + 1
        self.btn = [[0 for x in range( 2 *self.nb_columns)] for x in range(self.nb_flights)]
        self.curr_plots = 0

        for idx, tf_curr in enumerate(text_fid_mat):
            curr_column = idx /30
            curr_row = idx - curr_column * 30

            text_curr, fid_curr = tf_curr
            curr_column_used = 2* curr_column
            self.btn[curr_row][curr_column_used + 1] = Button(self
                                                              , text=fid_curr
                                                              , command=lambda x1=curr_row, y1=curr_column:
                self.display_graph(x1, y1))
            self.btn[curr_row][curr_column_used] = Label(self, text=text_curr)

            self.btn[curr_row][curr_column_used + 1].grid(column=curr_column_used + 1, row=curr_row)
            self.btn[curr_row][curr_column_used].grid(column=curr_column_used, row=curr_row)

    def display_graph(self, x1, y1):

        if self.curr_plots == 0:
            self.curr_plots = 1

        else:  # close the other one, display new one
            plt.close()

        flight_idx_compute = 30 * y1 + x1
        plot_from_flight_id(flight_id=self.text_fid_mat[flight_idx_compute][1])
