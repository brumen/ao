# these two functions (d_v_fct and s_v_fct) are here for pickle reasons
def s_v_fct(s, t):
    """
    volatility structure of the model,

    :param s: volatility at time t
    :type s:  double
    :param t: time at which volatility is evaluted
    :type t:  double
    :returns: volatility of the model
    :rtype:   double
    """

    return s


def d_v_fct(d, t):
    """
    drift structure of the model,

    :param d: drift at time t
    :type d:  double
    :param t: time at which drift is evaluted
    :type t:  double
    :returns: drift of the model
    :rtype:   double
    """

    return d
