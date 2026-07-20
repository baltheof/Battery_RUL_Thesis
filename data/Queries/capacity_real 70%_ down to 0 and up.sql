WITH BatteryCycles AS (
    SELECT
        tc.Battery_ID,
        tc.Cycle_Index,
        tc.Capacity_Ah,
        b.Nominal_Capacity,
        ROW_NUMBER() OVER (
            PARTITION BY tc.Battery_ID
            ORDER BY tc.Cycle_Index DESC
        ) AS Reverse_Row_Number
    FROM dbo.TEST_CYCLES AS tc
    INNER JOIN dbo.BATTERIES AS b
        ON tc.Battery_ID = b.Battery_ID
    WHERE tc.Operation_Type = 'discharge'
)
SELECT
    Battery_ID,
    MAX(Nominal_Capacity) AS Nominal_Capacity,
    MAX(Nominal_Capacity) * 0.70 AS Failure_Threshold,
    MIN(Capacity_Ah) AS Minimum_Capacity,
    MIN(
        CASE
            WHEN Capacity_Ah < Nominal_Capacity * 0.70
            THEN Cycle_Index
        END
    ) AS First_Cycle_Below_Threshold,
    SUM(
        CASE
            WHEN Capacity_Ah < Nominal_Capacity * 0.70
            THEN 1
            ELSE 0
        END
    ) AS Cycles_Below_Threshold,
    MAX(Cycle_Index) AS Last_Cycle,
    MAX(
        CASE
            WHEN Reverse_Row_Number = 1
            THEN Capacity_Ah
        END
    ) AS Last_Capacity
FROM BatteryCycles
GROUP BY Battery_ID
ORDER BY Battery_ID;