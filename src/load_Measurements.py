import os
import glob
import pandas as pd
from db_connection import get_engine # Βάλε το σωστό όνομα αν διαφέρει

def load_Measurements():
    engine = get_engine()
    if engine is None: return

    # 1. Path για τον ΦΑΚΕΛΟ με τα Raw δεδομένα 
    raw_folder_path = r'C:/Users/balth/Desktop/Battery_RUL_Thesis/data/cleaned_dataset/data/raw'

    # Μαζεύουμε όλα τα αρχεία που τελειώνουν σε .csv
    all_csv_files = glob.glob(os.path.join(raw_folder_path, "*.csv"))

    if not all_csv_files:
        print("Not found csv in this directory..")
        return
    
    print(f"📂 Found {len(all_csv_files)} files to load. Starting the process...")

    # 2. ΕΝΑΡΞΗ ΤΗΣ ΛΟΥΠΑΣ
    for file_path in all_csv_files:
        file_name = os.path.basename(file_path)
        print(f"⏳ Processing and loading file: {file_name}...")

        try:
            # --- Η ΜΑΓΕΙΑ ΣΥΜΒΑΙΝΕΙ ΕΔΩ ---
            # Εξαγωγή του Cycle_ID από το όνομα (π.χ. '07062.csv' -> 7062)
            cycle_id = int(file_name.replace('.csv', ''))

            # Διαβάζουμε το μικρό αρχείο CSV
            raw_df = pd.read_csv(file_path)

            # ΑΣΦΑΛΕΙΑ: Αν το CSV είναι άδειο
            if len(raw_df) == 0:
                print(f"  -> ⚠️ Το αρχείο είναι άδειο. Προσπέραση...")
                continue

            # Προσθέτουμε χειροκίνητα τη στήλη Cycle_ID σε όλες τις γραμμές του αρχείου!
            raw_df['Cycle_ID'] = cycle_id
            # -------------------------------

            # 3. Επιλογή και Μετονομασία στηλών (Ακριβώς όπως στο Excel σου)
            upload_df = raw_df[[
                'Cycle_ID',
                'Voltage_measured', 
                'Current_measured', 
                'Temperature_measured', 
                'Time'
            ]].copy()

            upload_df.columns = [
                'Cycle_ID', 
                'Voltage_Measured', 
                'Current_Measured', 
                'Temperature_Measure', 
                'Time_Seconds'
            ]

            # 4. Φόρτωση στην SQL
            upload_df.to_sql('MEASUREMENTS', con=engine, if_exists='append', index=False, chunksize=500)
            print(f"  -> ✅ Success: {file_name} loaded into Cycle_ID {cycle_id}!")

        except Exception as e:
            # Πιάνει τα λάθη για το μεμονωμένο αρχείο και συνεχίζει
            print(f"  -> ❌ ERROR IN {file_name} : {e}")

if __name__ == "__main__":
    load_Measurements()