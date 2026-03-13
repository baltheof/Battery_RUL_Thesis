import pandas as pd
from db_connection import get_engine


#-----------DATA LOADING - BRANCH ------------

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

#--------------------------------------------------


#----------LOAD TEST CYCLES - BRANCH --------------

def load_test_cycles():
    engine = get_engine()
    if engine is None: return

    file_path = r'C:/Users/balth/Desktop/Battery_RUL_Thesis/data/cleaned_dataset/metadata.csv'

    try:
        df = pd.read_csv(file_path)

        # 1. Επιλογή στηλών
        cycles_data = df[['battery_id', 'test_id', 'type', 'Capacity']].copy()
        cycles_data.columns = ['Battery_ID', 'Cycle_Index', 'Operation_Type', 'Capacity_Ah']

        # 2. ΕΞΑΝΑΓΚΑΣΜΟΣ ΤΥΠΩΝ (Το κλειδί για τη λύση του σφάλματος 8114)
        # Μετατρέπουμε το Capacity_Ah σε float και το Cycle_Index σε int
        cycles_data['Capacity_Ah'] = pd.to_numeric(cycles_data['Capacity_Ah'], errors='coerce')
        cycles_data['Cycle_Index'] = pd.to_numeric(cycles_data['Cycle_Index'], errors='coerce').astype('Int64')

        # 3. Καθαρισμός: Αφαιρούμε γραμμές που έγιναν NaN (π.χ. αν υπήρχε κείμενο αντί για νούμερο)
        cycles_data = cycles_data.dropna(subset=['Battery_ID', 'Capacity_Ah', 'Cycle_Index'])

        # 4. Φόρτωση στην SQL
        cycles_data.to_sql(
            'TEST_CYCLES', 
            con=engine, 
            if_exists='append', 
            index=False, 
            chunksize=100 
        )

        print(f" Success: {len(cycles_data)} rows loaded into TEST_CYCLES!")

    except Exception as e:
        # Χρησιμοποιούμε μόνο το 'e' εδώ για να αποφύγουμε το NameError
        print(f" LOADING ERROR: {e}")

if __name__ == "__main__":
    # 1. Πρώτα ελέγχουμε/φορτώνουμε τις μπαταρίες
    load_battery_metadata()
    
    # 2. ΜΕΤΑ φορτώνουμε τους κύκλους (ΠΡΟΣΕΞΕ ΝΑ ΜΗΝ ΕΙΝΑΙ ΣΕ ΣΧΟΛΙΟ #)
    load_test_cycles()

# --------------------------------------------------