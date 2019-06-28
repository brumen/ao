import mysql.connector
import ao_codes
# DB_HOST is where the params for pricing are held, DB_HOST_CALIBRATE is where the total db is held
from ao_codes import DB_HOST, DB_HOST_CALIBRATE, DATABASE, DB_USER


class MysqlConnectorEnv(object):

    def __init__( self
                , calibrate_db = False):
        """
        :param calibrate_db: indicator whether to use the database for calibration or not
                             (default = not, use database for service on odroid.local)
        """

        self._calibrate_db = calibrate_db

    def __enter__(self):
        self.connection = mysql.connector.connect( host     = DB_HOST if not self._calibrate_db else DB_HOST_CALIBRATE
                                                 , database = DATABASE
                                                 , user     = DB_USER
                                                 , password = ao_codes.brumen_mysql_pass )
        return self.connection

    def __exit__(self, *args):  # TODO: CHECK IF THIS IS CORRECT
        self.connection.close()


def make_pymysql_conn(calibrate_db = False):
    """
    Making a PyMysql connection
    """

    return mysql.connector.connect( host     = DB_HOST if not calibrate_db else DB_HOST_CALIBRATE
                                  , database = DATABASE
                                  , user     = DB_USER
                                  , password = ao_codes.brumen_mysql_pass )