import pandas as pd
import numpy as np
from db_connection import get_engine
from sqlalchemy import text

def clean_zero_capacities():
    engine = get_engine()
    if engine is None : return

    print("Sending targeted UPDATE command to the database...")

    update_query = text("""
        WITH CalculatedCapacities AS (
            SELECT 
                Cycle_ID,
                LAG(Capacity_Ah) OVER (PARTITION BY Battery_ID ORDER BY Cycle_Index) AS Prev_Capacity,
                LEAD(Capacity_Ah) OVER (PARTITION BY Battery_ID ORDER BY Cycle_Index) AS Next_Capacity
            FROM TEST_CYCLES
        )
        UPDATE TEST_CYCLES
        SET Capacity_Ah = (C.Prev_Capacity + C.Next_Capacity) / 2.0
        FROM TEST_CYCLES
        INNER JOIN CalculatedCapacities C ON TEST_CYCLES.Cycle_ID = C.Cycle_ID
        WHERE TEST_CYCLES.Capacity_Ah = 0
          AND C.Prev_Capacity IS NOT NULL 
          AND C.Next_Capacity IS NOT NULL;
    """)

    # Execute the query safely 
    with engine.begin() as conn:
        conn.execute(update_query)

    print("Correction completed successfully!")

if __name__ == "__main__":
    clean_zero_capacities()