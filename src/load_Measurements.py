import os
import pandas as pd
from db_connection import get_engine 
def load_Measurements_Correctly():
    engine = get_engine()
    if engine is None: return

    # 1. Διαβάζουμε το "Λεξικό"
    metadata_path = r'C:/Users/balth/Desktop/Battery_RUL_Thesis/data/cleaned_dataset/metadata.csv' 
    metadata_df = pd.read_csv(metadata_path)

    # 2. Κρατάμε ΑΥΣΤΗΡΑ μόνο τις αποφορτίσεις (discharges)
    discharges_df = metadata_df[metadata_df['type'] == 'discharge'].copy()

    # 3. Κατεβάζουμε τα σωστά ID από τη βάση δεδομένων
    sql_query = "SELECT Cycle_ID, Battery_ID, Cycle_Index FROM TEST_CYCLES"
    test_cycles_df = pd.read_sql(sql_query, engine)

    # 4. Παντρεύουμε (Merge) το Λεξικό με τη βάση
    # Τώρα ξέρουμε ΑΚΡΙΒΩΣ ποιο filename πάει σε ποιο Cycle_ID!
    mapping_df = pd.merge(
        discharges_df,
        test_cycles_df,
        left_on=['battery_id', 'test_id'],
        right_on=['Battery_ID', 'Cycle_Index']
    )

    # Ο φάκελος με τα μικρά CSV
    raw_folder_path = r'C:/Users/balth/Desktop/Battery_RUL_Thesis/data/cleaned_dataset/data/raw'

    print(f"Found exactly {len(mapping_df)} discharge files! Starting the loading process...")

    # 5. Η Λούπα φόρτωσης
    for index, row in mapping_df.iterrows():
        file_name = row['filename']
        cycle_id = row['Cycle_ID']
        
        file_path = os.path.join(raw_folder_path, file_name)
        
        print(f" Reading {file_name} (maps to Cycle_ID {cycle_id})...")

        try:
            # Διαβάζουμε το μικρό CSV
            raw_df = pd.read_csv(file_path)

            if len(raw_df) == 0:
                continue

            # Του κολλάμε το ΑΠΟΛΥΤΑ ΣΩΣΤΟ Cycle_ID
            raw_df['Cycle_ID'] = cycle_id

            # Μετονομασία στηλών για την SQL
            upload_df = raw_df[[
                'Cycle_ID', 'Voltage_measured', 'Current_measured', 'Temperature_measured', 'Time'
            ]].copy()

            upload_df.columns = [
                'Cycle_ID', 'Voltage_Measured', 'Current_Measured', 'Temperature_Measure', 'Time_Seconds'
            ]

            # Ανέβασμα στη βάση
            upload_df.to_sql('MEASUREMENTS', con=engine, if_exists='append', index=False, chunksize=500)
            print(f"  Uploaded successfully!")

        except Exception as e:
            print(f"  ERROR IN FILE {file_name} : {e}")

if __name__ == "__main__":
    load_Measurements_Correctly()