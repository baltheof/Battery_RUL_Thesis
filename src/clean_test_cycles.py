import pandas as pd
import numpy as np
from sqlalchemy import text
from db_connection import get_engine

def clean_consecutive_outliers():
    engine = get_engine()
    if engine is None: 
        return

    print("Fetching discharge cycles from database...")
    query = """
        SELECT Cycle_ID, Battery_ID, Cycle_Index, Capacity_Ah 
        FROM TEST_CYCLES 
        WHERE Operation_Type = 'discharge' 
        ORDER BY Battery_ID, Cycle_Index
    """
    df = pd.read_sql(query, engine)

    # 1. Keep original values to compare later
    original_capacities = df['Capacity_Ah'].copy()

    print("Identifying bad values (< 0.8 Ah) and applying Pandas Interpolation...")
    
    # 2. Identify Anomalies: Turn ALL values below 0.8 Ah into NaN (empty)
    # This catches blocks of continuous errors (like 0.057)
    df.loc[df['Capacity_Ah'] < 0.8, 'Capacity_Ah'] = np.nan

    # 3. Interpolation: Pandas fills the gaps smoothly across each battery
    df['Capacity_Ah'] = df.groupby('Battery_ID')['Capacity_Ah'].transform(lambda x: x.interpolate(method='linear'))
    
    # Optional: If the very first or last row of a battery was NaN, fill it
    df['Capacity_Ah'] = df.groupby('Battery_ID')['Capacity_Ah'].transform(lambda x: x.ffill().bfill())

    # 4. Find exactly which rows were modified
    changed_rows = df[df['Capacity_Ah'] != original_capacities].dropna(subset=['Capacity_Ah'])
    issues_found = len(changed_rows)
    
    print(f"Found {issues_found} anomalous consecutive values. Updating database safely...")

    if issues_found == 0:
        print("No anomalies found. Database is clean!")
        return

    # 5. Database Update: Update ONLY the specific rows that were fixed
    with engine.begin() as conn:
        for _, row in changed_rows.iterrows():
            stmt = text("""
                UPDATE TEST_CYCLES 
                SET Capacity_Ah = :cap 
                WHERE Cycle_ID = :cid
            """)
            conn.execute(stmt, {"cap": row['Capacity_Ah'], "cid": row['Cycle_ID']})

    print("Correction of consecutive anomalies completed successfully!")

if __name__ == "__main__":
    clean_consecutive_outliers()