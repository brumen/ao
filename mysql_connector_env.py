import mysql.connector

# DB_HOST is where the params for pricing are held, DB_HOST_CALIBRATE is where the total db is held
from ao_codes import DB_HOST, DATABASE, DB_USER


class MysqlConnectorEnv:

    def __init__( self
                , host : str     = DB_HOST
                , database : str = DATABASE
                , user     : str = DB_USER
                , password : str = None ):
        """ Initializes the MySQL connector environment.

        :param host: host mysql connection
        :param database: database where to connect
        :param user: user used for connection
        :param password: potential password used.
        """

        self.host     = host
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
