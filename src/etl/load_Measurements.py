import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_connection import get_engine

def load_Measurements_Correctly():
    engine = get_engine()
    if engine is None: return

    # 1. Read the metadata dictionary
    metadata_path = r'C:/Users/balth/Desktop/Battery_RUL_Thesis/data/cleaned_dataset/metadata.csv' 
    metadata_df = pd.read_csv(metadata_path)

    # 2. Strictly keep only discharge cycles
    discharges_df = metadata_df[metadata_df['type'] == 'discharge'].copy()

    # Create the corresponding counter (Match_Index) in the metadata
    discharges_df = discharges_df.sort_values(['battery_id', 'test_id'])
    discharges_df['Match_Index'] = discharges_df.groupby('battery_id').cumcount() + 1

    # 3. Fetch the correct IDs from the database
    sql_query = "SELECT Cycle_ID, Battery_ID, Cycle_Index FROM TEST_CYCLES"
    test_cycles_df = pd.read_sql(sql_query, engine)

    # 4. Merge the metadata with the database records
    mapping_df = pd.merge(
        discharges_df,
        test_cycles_df,
        left_on=['battery_id', 'Match_Index'],
        right_on=['Battery_ID', 'Cycle_Index']
    )

    # Folder containing the raw CSV files
    raw_folder_path = r'C:/Users/balth/Desktop/Battery_RUL_Thesis/data/cleaned_dataset/data/raw'

    print(f"Found exactly {len(mapping_df)} discharge files! Starting the loading process...")

    # 5. Loading loop
    for index, row in mapping_df.iterrows():
        file_name = row['filename']
        cycle_id = row['Cycle_ID']
        battery_id = row['Battery_ID'] # Extract the battery name
        
        file_path = os.path.join(raw_folder_path, file_name)
        
        print(f" Reading {file_name} (maps to Cycle_ID {cycle_id}, Battery {battery_id})...")

        try:
            # Read the raw CSV file
            raw_df = pd.read_csv(file_path)

            if len(raw_df) == 0:
                continue

            # Append Cycle_ID and Battery_ID
            raw_df['Cycle_ID'] = cycle_id
            raw_df['Battery_ID'] = battery_id # Add the new column

            # Rename columns for SQL schema
            upload_df = raw_df[[
                'Cycle_ID', 'Battery_ID', 'Voltage_measured', 'Current_measured', 'Temperature_measured', 'Time'
            ]].copy()

            upload_df.columns = [
                'Cycle_ID', 'Battery_ID', 'Voltage_Measured', 'Current_Measured', 'Temperature_Measure', 'Time_Seconds'
            ]

            # Upload to database
            upload_df.to_sql('MEASUREMENTS', con=engine, if_exists='append', index=False, chunksize=500)
            print(f"  Uploaded successfully!")

        except Exception as e:
            print(f"  ERROR IN FILE {file_name} : {e}")

if __name__ == "__main__":
    load_Measurements_Correctly()