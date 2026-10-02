CREATE OR ALTER VIEW BATTERY_STATUS AS
WITH last_valid AS (
    SELECT a.Battery_ID, a.Cycle_Index AS Last_Cycle,
           a.Capacity_Ah AS Last_Cap, a.SoH AS Last_SoH
    FROM CYCLE_FEATURES_ALL AS a
    INNER JOIN (
        SELECT Battery_ID, MAX(Cycle_Index) AS Mx
        FROM CYCLE_FEATURES_ALL
        WHERE Flag = 1
        GROUP BY Battery_ID
    ) AS m ON a.Battery_ID = m.Battery_ID AND a.Cycle_Index = m.Mx
),
stats AS (
    SELECT Battery_ID,
           COUNT(*) AS Total_Cycles,
           SUM(CASE WHEN Flag = 1 THEN 1 ELSE 0 END) AS Valid_Cycles,
           SUM(CASE WHEN Flag = 1 AND SoH > 0.75 THEN 1 ELSE 0 END) AS Healthy_Cycles,
           MAX(Nominal) AS Nominal,
           MAX(Is_Censored) AS Is_Censored,
           MAX(CASE WHEN RUL IS NOT NULL THEN 1 ELSE 0 END) AS Has_RUL
    FROM CYCLE_FEATURES_ALL
    GROUP BY Battery_ID
)
SELECT s.Battery_ID, s.Total_Cycles, s.Valid_Cycles, s.Healthy_Cycles,
       ROUND(s.Nominal, 3) AS Nominal,
       ROUND(s.Nominal * 0.7, 3) AS Fail_Thr,
       ROUND(l.Last_Cap, 3) AS Last_Cap,
       ROUND(l.Last_SoH, 3) AS Last_SoH,
       CASE
         WHEN s.Has_RUL = 1 AND s.Is_Censored = 0 THEN 'CONFIRMED'
         WHEN s.Has_RUL = 1 AND s.Is_Censored = 1 THEN 'CENSORED'
         WHEN s.Valid_Cycles < 20 THEN 'OUT: FEW_CYCLES'
         WHEN s.Healthy_Cycles = 0 THEN 'OUT: NEVER_HEALTHY'
         WHEN l.Last_SoH >= 0.80 THEN 'OUT: LAST_SOH_HIGH'
         ELSE 'OUT: OTHER'
       END AS Status
FROM stats AS s
INNER JOIN last_valid AS l ON s.Battery_ID = l.Battery_ID