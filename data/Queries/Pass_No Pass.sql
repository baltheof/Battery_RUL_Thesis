WITH battery_stats AS (
    SELECT DISTINCT
        Battery_ID,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY Capacity_Ah) 
            OVER (PARTITION BY Battery_ID) AS Nominal_95pct
    FROM TEST_CYCLES
    WHERE Operation_Type = 'discharge'
      AND Capacity_Ah > 0.5
),
cycles_with_threshold AS (
    SELECT 
        tc.Battery_ID,
        tc.Cycle_Index,
        tc.Capacity_Ah,
        bs.Nominal_95pct,
        bs.Nominal_95pct * 0.75 AS threshold,
        CASE WHEN tc.Capacity_Ah < bs.Nominal_95pct * 0.75 
             THEN 1 ELSE 0 END AS below_threshold
    FROM TEST_CYCLES tc
    INNER JOIN battery_stats bs ON tc.Battery_ID = bs.Battery_ID
    WHERE tc.Operation_Type = 'discharge'
      AND tc.Capacity_Ah > 0.5
),
consecutive_check AS (
    SELECT
        Battery_ID,
        Cycle_Index,
        below_threshold,
        -- Αθροίζει τους 3 τελευταίους κύκλους
        SUM(below_threshold) OVER (
            PARTITION BY Battery_ID 
            ORDER BY Cycle_Index 
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS consecutive_below
    FROM cycles_with_threshold
),
battery_failure AS (
    SELECT DISTINCT Battery_ID
    FROM consecutive_check
    WHERE consecutive_below = 3
)
SELECT 
    b.Battery_ID,
    b.Nominal_Capacity,
    ROUND(bs.Nominal_95pct, 3) AS Nominal_95pct,
    ROUND(bs.Nominal_95pct * 0.75, 3) AS Threshold_75pct,
    COUNT(tc.Cycle_ID) AS Total_Cycles,
    ROUND(MIN(tc.Capacity_Ah), 3) AS Min_Capacity,
    ROUND(MAX(tc.Capacity_Ah), 3) AS Max_Capacity,
    MAX(tc.Cycle_Index) AS Last_Cycle,
    CASE 
        WHEN bf.Battery_ID IS NOT NULL
        THEN 'PASSED'
        ELSE 'NOT PASSED'
    END AS Status
FROM BATTERIES b
INNER JOIN TEST_CYCLES tc ON b.Battery_ID = tc.Battery_ID
INNER JOIN battery_stats bs ON b.Battery_ID = bs.Battery_ID
LEFT JOIN battery_failure bf ON b.Battery_ID = bf.Battery_ID
WHERE tc.Operation_Type = 'discharge'
  AND tc.Capacity_Ah > 0.5
GROUP BY b.Battery_ID, b.Nominal_Capacity, bs.Nominal_95pct, bf.Battery_ID
ORDER BY Status DESC, b.Battery_ID