import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_connection import get_engine

#---------- DATA LOADING - BRANCH ------------

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

        print("The battery data have been succesfully loaded into the SQL Database")

    except Exception as e :
        #Handle Primary Key violation if data already exists
        if "Violation of PRIMARY KEY" in str(e):
            print("Battery data are already in the sql database")
        else:
            print(f"Loading ERROR: {e}")

if __name__ == "__main__":
    load_battery_metadata()

#--------------------------------------------------


#----------LOAD TEST CYCLES - BRANCH --------------

def load_test_cycles():

    # Initialize database connection
    engine = get_engine()
    if engine is None: return #stop the compile

    file_path = r'C:/Users/balth/Desktop/Battery_RUL_Thesis/data/cleaned_dataset/metadata.csv'

    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # 1. Select specific columns and rename them to match the SQL schema
        cycles_data = df[['battery_id', 'test_id', 'type', 'Capacity']].copy()
        cycles_data.columns = ['Battery_ID', 'Cycle_Index', 'Operation_Type', 'Capacity_Ah']

        # 2. DATA TYPE CONVERSION (Fixes SQL Error 8114)
        cycles_data['Capacity_Ah'] = pd.to_numeric(cycles_data['Capacity_Ah'], errors='coerce')
        cycles_data['Cycle_Index'] = pd.to_numeric(cycles_data['Cycle_Index'], errors='coerce').astype('Int64')

        # 3. CLEANING: Remove any rows that became NaN during conversion
        cycles_data = cycles_data.dropna(subset=['Battery_ID', 'Capacity_Ah', 'Cycle_Index'])

        # --- ΝΕΑ ΠΡΟΣΘΗΚΗ: ΑΥΤΟΜΑΤΗ ΑΝΑΡΙΘΜΗΣΗ (ΞΕΚΙΝΑΕΙ ΑΠΟ ΤΟ 1) ---
        
        # Ταξινομούμε τα δεδομένα πρώτα ανά μπαταρία και μετά με τον παλιό αριθμό για να κρατήσουμε τη σωστή χρονική σειρά
        cycles_data = cycles_data.sort_values(['Battery_ID', 'Cycle_Index'])
        
        # Διαγράφουμε τον παλιό ακατάστατο αριθμό και φτιάχνουμε έναν νέο, σειριακό μετρητή που ξεκινάει από το 1
        cycles_data['Cycle_Index'] = cycles_data.groupby('Battery_ID').cumcount() + 1
        
        # ---------------------------------------------------------------

        # 4. UPLOAD TO SQL DATABASE
        # Use chunksize to avoid the 2100 parameter limit (Error gkpj)
        cycles_data.to_sql(
            'TEST_CYCLES', 
            con=engine, 
            if_exists='append', 
            index=False, 
            chunksize=100 
        )

        print(f"Success: {len(cycles_data)} rows loaded into TEST_CYCLES with sequential Cycle_Index!")

    except Exception as e:
        # Print error details for debugging
        print(f"LOADING ERROR: {e}")

if __name__ == "__main__":
    # 1. First, we check/load the main battery profiles (Parent Table)
    load_battery_metadata()
    
    # 2. THEN, we load the historical test cycles (Child Table)
    # These records depend on the batteries existing in the database.
    load_test_cycles()

# ------------------------------------------------