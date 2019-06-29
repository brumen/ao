import mysql.connector
import ao_codes
# DB_HOST is where the params for pricing are held, DB_HOST_CALIBRATE is where the total db is held
from ao_codes import DB_HOST, DB_HOST_CALIBRATE, DATABASE, DB_USER


class MysqlConnectorEnv(object):

    def __init__( self
                , host     = DB_HOST
                , database = DATABASE
                , user     = DB_USER
                , password = ao_codes.brumen_mysql_pass ):
        """
        :param host: host mysql connection
        :param database: database where to connect
        :param user: user used for connection
        :param password: potential password used.
        """

        self.host = host
        self.database = database
        self.user     = user
        self.password = password

    def __enter__(self):
        self.connection = mysql.connector.connect( host     = self.host
                                                 , database = self.database
                                                 , user     = self.user
                                                 , password = self.password )
        return self.connection

    def __exit__(self, *args):
        self.connection.close()
