import sqlalchemy
import urllib

def get_engine():

    server = r'(local)\SQLEXPRESS'
    database = 'BATTERY_DB'

    params = urllib.parse.quote_plus(
        f'DRIVER={{ODBC Driver 17 for SQL Server}};'
        f'SERVER={server};'
        f'DATABASE={database};'
        f'Trusted_Connection=yes;'
    )

    connection_url = f"mssql+pyodbc:///?odbc_connect={params}"

    try:
        engine = sqlalchemy.create_engine(connection_url)
        with engine.connect() as conn:
            print("Connection with SQL Server success")
        return engine
    except Exception as e:
        print(f"Connection Failed {e}")
        return None
    
if __name__ == "__main__":
    get_engine()