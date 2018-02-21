# finds the relevant carriers
import getpass  # for username
import pandas as pd

from ao_scripts.get_data import validate_airport

from ao_codes            import iata_codes_airlines, error_log
from mysql_connector_env import MysqlConnectorEnv


log_file = error_log + '_' + getpass.getuser()


def get_carrier_l(origin, dest):
    """
    gets the list from the params database

    """
    # get the three letter codes from origin, dest
    origin_upper, origin_valid = validate_airport(origin)
    dest_upper, dest_valid = validate_airport(dest)

    if origin_valid and dest_valid:  # return carriers

        exec_q = """
        SELECT DISTINCT(carrier) 
        FROM params 
        WHERE orig = '{0}' AND dest = '{1}'
        """.format(origin_upper, dest_upper)

        with MysqlConnectorEnv() as mconn:
            df1 = pd.read_sql_query(exec_q, mconn)

        ret_cand_1 = list(df1['carrier'])
        if len(ret_cand_1) == 0:  # no result
            return False, []
        else:  # we have to return all the candidates
            # extend with flight names
            ret_cand_1.extend([iata_codes_airlines[x] for x in ret_cand_1])
            return True, ret_cand_1
    else:  # wrong inputs
        return False, []
