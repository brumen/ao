import mysql.connector
import ao_codes
from   ao_codes            import DB_HOST, DATABASE, DB_USER


class MysqlConnectorEnv(object):

    def __enter__(self):
        self.connection = mysql.connector.connect( host     = DB_HOST
                                                 , database = DATABASE
                                                 , user     = DB_USER
                                                 , password = ao_codes.brumen_mysql_pass )
        return self.connection

    def __exit__(self, *args):  # TODO: CHECK IF THIS IS CORRECT
        self.connection.close()
