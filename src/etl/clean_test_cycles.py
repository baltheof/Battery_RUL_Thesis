import pandas as pd
import numpy as np
from sqlalchemy import text
from db_connection import get_engine
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def fix_only_zeros():
    engine = get_engine()
    if engine is None: 
        return

    print("Fetching discharge cycles from database...")
    # Loading only discharge cycles to process the zeros
    query = """
        SELECT Cycle_ID, Battery_ID, Cycle_Index, Capacity_Ah 
        FROM TEST_CYCLES 
        WHERE Operation_Type = 'discharge' 
        ORDER BY Battery_ID, Cycle_Index
    """
    df = pd.read_sql(query, engine)

    # 1. Identity Check: Find exactly where Capacity is 0
    # We create a mask for values that are exactly 0
    zeros_mask = (df['Capacity_Ah'] == 0)
    
    num_zeros = zeros_mask.sum()
    print(f"Found {num_zeros} absolute zero values.")

    if num_zeros == 0:
        print("No zero values found. Database is already consistent with Excel.")
        return

    # 2. Preparation for Interpolation
    # We replace ONLY the zeros with NaN so Pandas can fill them
    df.loc[zeros_mask, 'Capacity_Ah'] = np.nan

    print("Applying Linear Interpolation for zero values...")
    # Group by battery to ensure we don't mix data between different batteries
    df['Capacity_Ah'] = df.groupby('Battery_ID')['Capacity_Ah'].transform(lambda x: x.interpolate(method='linear'))
    
    # Handle cases where a zero might be at the very start or end
    df['Capacity_Ah'] = df.groupby('Battery_ID')['Capacity_Ah'].transform(lambda x: x.ffill().bfill())

    # 3. Targeted Database Update
    # We only want to update the rows that were originally 0
    to_update = df[zeros_mask].copy()

    print(f"Updating {len(to_update)} records in the database...")
    
    with engine.begin() as conn:
        for _, row in to_update.iterrows():
            stmt = text("""
                UPDATE TEST_CYCLES 
                SET Capacity_Ah = :cap 
                WHERE Cycle_ID = :cid
            """)
            conn.execute(stmt, {"cap": row['Capacity_Ah'], "cid": row['Cycle_ID']})

    print("Strict cleanup completed successfully! Only zero values were modified.")

if __name__ == "__main__":
    fix_only_zeros()