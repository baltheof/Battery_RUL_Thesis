import pandas as pd
from db_connection import get_engine

def load_battery_metadata():
    
    # 1. Get database connection- previous file
    engine = get_engine()
    if engine is None:
        return
    
    # 2. Define file path
    file_path = r'C:/Users/balth/Desktop/Battery_RUL_Thesis/data/cleaned_dataset/metadata.csv'

    try:
        # Read CSV into DataFrame
        df = pd.read_csv(file_path)

        # Filter for 'discharge' cycles only
        discharge_only_df = df[df['type'] == 'discharge']

        # Keep specific columns and get the first discharge capacity per battery
        battery_data = discharge_only_df [['battery_id','Capacity']].drop_duplicates(subset=['battery_id']).copy()
        
        #Drop rows with missing battery IDs
        battery_data = battery_data.dropna(subset=['battery_id'])

        # Add source dataset and rename columns to match SQL schema
        battery_data['Source_Dataset'] = 'NASA'
        battery_data.columns = ['Battery_ID', 'Nominal_Capacity', 'Source_Dataset']

        # Load data into SQL database
        battery_data.to_sql('BATTERIES', con=engine, if_exists='append', index=False)

        print(" The battery data have been succesfully loaded into the SQL Database")

    except Exception as e :
        #Handle Primary Key violation if data already exists
        if "Violation of PRIMARY KEY" in str(e):
            print("battery data are already in the sql database")
        else:
            print(f" Loading ERROR : {e} ")

if __name__ == "__main__":
    load_battery_metadata()