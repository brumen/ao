

def test_only_1():
    direct_flights = """
    SELECT DISTINCT as_of, orig, dest, dep_date, arr_date, price 
    FROM flights WHERE orig= 'SFO' AND dest = 'EWR' AND carrier='UA' 
    ORDER BY as_of"""

    dep_dates_str = """
    SELECT DISTINCT dep_date FROM flights WHERE orig= 'SFO' AND dest = 'EWR' AND carrier='UA'
    """
    dep_dates = pd.read_sql_query(dep_dates_str, ao_db.conn_ao,
                                  parse_dates={'dep_date': '%Y-%m-%dT%H:%M:%S'})

    df1 = pd.read_sql_query(direct_flights, ao_db.conn_ao,
                            parse_dates={'as_of': '%Y-%m-%d',
                                         'dep_date': '%Y-%m-%dT%H:%M:%S',
                                         'arr_date': '%Y-%m-%dT%H:%M:%S'})

    return df1, dep_dates
