import pandas as pd
from db_connection import get_engine

def load_battery_metadata():
    
    # 1. we take the connection from previous file
    engine = get_engine()
    if engine is None:
        return
    
    # 2. file metadata.csv
    file_path = r'C:/Users/balth/Desktop/Battery_RUL_Thesis/data/cleaned_dataset/metadata.csv'

    try:
        # Reading the file
        df = pd.read_csv(file_path)

        # 1. Κρατάμε ΜΟΝΟ τις στήλες που θέλουμε
        # 2. Χρησιμοποιούμε το drop_duplicates για να μείνει κάθε Battery_ID ΜΙΑ φορά
        battery_data = df[['battery_id', 'Capacity']].drop_duplicates(subset=['battery_id']).copy()
        
        # 3. Φιλτράρουμε τυχόν κενές τιμές (αν υπάρχουν)
        battery_data = battery_data.dropna(subset=['battery_id'])

        battery_data['Source_Dataset'] = 'NASA'
        battery_data.columns = ['Battery_ID', 'Nominal_Capacity', 'Source_Dataset']

        # 4. Ανέβασμα στην SQL
        battery_data.to_sql('BATTERIES', con=engine, if_exists='append', index=False)

        # 3. Ανέβασμα στην SQL
        battery_data.to_sql('BATTERIES', con=engine, if_exists='append', index=False)

        print(" The battery data have been succesfully loaded into the SQL Database")

    except Exception as e :
        # Αν το σφάλμα είναι "Duplicate Key", σημαίνει ότι οι μπαταρίες είναι ήδη μέσα
        if "Violation of PRIMARY KEY" in str(e):
            print("battery data are already in the sql database")
        else:
            print(f" Loading ERROR : {e} ")

if __name__ == "__main__":
    load_battery_metadata()