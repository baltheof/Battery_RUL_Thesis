import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from sqlalchemy import text
from db_connection import get_engine

FAILURE_THRESHOLD_RATIO = 0.70

def extract_features():
    engine = get_engine()
    if engine is None:
        return

    print("Step 1/4: Loading discharge cycles from TEST_CYCLES...")
    cycles_query = """
        SELECT tc.Cycle_ID, tc.Battery_ID, tc.Cycle_Index, tc.Capacity_Ah,
               b.Nominal_Capacity
        FROM TEST_CYCLES tc
        JOIN BATTERIES b ON tc.Battery_ID = b.Battery_ID
        WHERE tc.Operation_Type = 'discharge'
        ORDER BY tc.Battery_ID, tc.Cycle_Index
    """
    cycles_df = pd.read_sql(cycles_query, engine)
    print(f"   Found {len(cycles_df)} discharge cycles across "
          f"{cycles_df['Battery_ID'].nunique()} batteries.")

    # Step 2: Compute RUL per battery
    print("Step 2/4: Computing RUL for each cycle...")

    rul_list = []
    
    # Group data by battery to safely compute RUL
    for battery_id, group in cycles_df.groupby('Battery_ID'):
        group = group.copy() # Prevent SettingWithCopyWarning
        threshold = group['Nominal_Capacity'].iloc[0] * FAILURE_THRESHOLD_RATIO
        failed = group[group['Capacity_Ah'] < threshold]

        if failed.empty:
            # Battery has not failed yet — RUL undefined, mark as NaN
            group['RUL'] = np.nan
        else:
            failure_cycle = failed['Cycle_Index'].iloc[0]
            group['RUL'] = failure_cycle - group['Cycle_Index']
            # Cycles after failure get RUL = 0
            group['RUL'] = group['RUL'].clip(lower=0)
            
        rul_list.append(group)

    # Recombine the processed groups
    cycles_df = pd.concat(rul_list, ignore_index=True)

    # Drop cycles with undefined RUL (batteries that never reached failure)
    before = len(cycles_df)
    cycles_df = cycles_df.dropna(subset=['RUL'])
    dropped = before - len(cycles_df)
    if dropped > 0:
        print(f"   Dropped {dropped} cycles from batteries with no observed failure.")
    print(f"   RUL computed for {len(cycles_df)} cycles.")

    # Step 3: Extract features from MEASUREMENTS
    print("Step 3/4: Extracting features from MEASUREMENTS...")

    cycle_ids = cycles_df['Cycle_ID'].tolist()

    # Load all relevant measurements in one query for efficiency
    ids_str = ','.join(str(i) for i in cycle_ids)
    meas_query = f"""
        SELECT Cycle_ID,
               MAX(Time_Seconds)           AS Discharge_Time,
               AVG(Temperature_Measure)    AS Temp_Mean,
               MAX(Temperature_Measure)    AS Temp_Max,
               MIN(Voltage_Measured)       AS Voltage_Min,
               AVG(Voltage_Measured)       AS Voltage_Mean,
               AVG(Current_Measured)       AS Current_Mean
        FROM MEASUREMENTS
        WHERE Cycle_ID IN ({ids_str}) AND Current_Measured < -0.1
        GROUP BY Cycle_ID
    """
    features_df = pd.read_sql(meas_query, engine)
    print(f"   Features extracted for {len(features_df)} cycles.")

    # Step 4: Merge and upload
    print("Step 4/4: Merging and uploading to CYCLE_FEATURES...")

    final_df = cycles_df.merge(features_df, on='Cycle_ID', how='inner')

    # Keep only the columns needed for ML
    final_df = final_df[[
    'Cycle_ID', 'Battery_ID', 'Cycle_Index', 'Capacity_Ah',
    'Discharge_Time', 'Temp_Mean', 'Temp_Max',
    'Voltage_Min', 'Voltage_Mean', 'Current_Mean',
    'RUL'
    ]]

    final_df['RUL'] = final_df['RUL'].astype(int)

    print(f"   Final dataset: {len(final_df)} rows, {final_df.shape[1]} columns.")
    print(f"   RUL range: {final_df['RUL'].min()} – {final_df['RUL'].max()} cycles")

    # Upload to SQL database
    final_df.to_sql(
        'CYCLE_FEATURES',
        con=engine,
        if_exists='replace',   # Use replace to allow safe re-runs
        index=False,
        chunksize=200
    )
    print("\nFeature extraction complete! Table CYCLE_FEATURES is ready.")

if __name__ == "__main__":
    extract_features()