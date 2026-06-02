import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_connection import get_engine

def load_battery_metadata():
    # Establish database connection
    engine = get_engine()
    if engine is None:
        return
    
    file_path = r'C:/Users/balth/Desktop/Battery_RUL_Thesis/data/cleaned_dataset/metadata.csv'

    try:
        df = pd.read_csv(file_path)

        # Filter for discharge cycles only
        discharge_only_df = df[df['type'] == 'discharge']

        # Keep specific columns and extract the first discharge capacity per battery
        battery_data = discharge_only_df[['battery_id', 'Capacity']].drop_duplicates(subset=['battery_id']).copy()
        
        # Drop rows with missing battery IDs
        battery_data = battery_data.dropna(subset=['battery_id'])

        # Add source dataset and align columns with SQL schema
        battery_data['Source_Dataset'] = 'NASA'
        battery_data.columns = ['Battery_ID', 'Nominal_Capacity', 'Source_Dataset']

        # Load data into SQL database
        battery_data.to_sql('BATTERIES', con=engine, if_exists='append', index=False)
        print("The battery data have been successfully loaded into the SQL database.")

    except Exception as e:
        # Handle Primary Key violation if data already exists
        if "Violation of PRIMARY KEY" in str(e):
            print("Battery data already exists in the SQL database.")
        else:
            print(f"Loading ERROR: {e}")

def load_test_cycles():
    # Establish database connection
    engine = get_engine()
    if engine is None: 
        return

    file_path = r'C:/Users/balth/Desktop/Battery_RUL_Thesis/data/cleaned_dataset/metadata.csv'

    try:
        df = pd.read_csv(file_path)

        # Select specific columns and align with SQL schema
        cycles_data = df[['battery_id', 'test_id', 'type', 'Capacity']].copy()
        cycles_data.columns = ['Battery_ID', 'Cycle_Index', 'Operation_Type', 'Capacity_Ah']

        # Convert data types to fix potential SQL errors
        cycles_data['Capacity_Ah'] = pd.to_numeric(cycles_data['Capacity_Ah'], errors='coerce')
        cycles_data['Cycle_Index'] = pd.to_numeric(cycles_data['Cycle_Index'], errors='coerce').astype('Int64')

        # Drop rows with missing critical values
        cycles_data = cycles_data.dropna(subset=['Battery_ID', 'Capacity_Ah', 'Cycle_Index'])

        # Sort data by battery and original index to maintain chronological order
        cycles_data = cycles_data.sort_values(['Battery_ID', 'Cycle_Index'])
        
        # Create a new sequential counter starting from 1
        cycles_data['Cycle_Index'] = cycles_data.groupby('Battery_ID').cumcount() + 1
        
        # Upload to SQL database in chunks to avoid parameter limits
        cycles_data.to_sql(
            'TEST_CYCLES', 
            con=engine, 
            if_exists='append', 
            index=False, 
            chunksize=100 
        )

        print(f"Success: {len(cycles_data)} rows loaded into TEST_CYCLES with sequential Cycle_Index!")

    except Exception as e:
        print(f"LOADING ERROR: {e}")

if __name__ == "__main__":
    # Check and load the main battery profiles (Parent Table)
    load_battery_metadata()
    
    # Load the historical test cycles (Child Table)
    load_test_cycles()