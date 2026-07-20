SELECT
    Battery_ID,
    MIN(CASE WHEN RUL = 0 THEN Cycle_Index END) AS First_Zero_Cycle,
    MAX(Cycle_Index) AS Last_Cycle,
    MIN(CASE WHEN RUL = 0 THEN Capacity_Ah END) AS Capacity_At_First_Zero,
    COUNT(CASE WHEN RUL = 0 THEN 1 END) AS Zero_RUL_Cycles
FROM CYCLE_FEATURES
GROUP BY Battery_ID
ORDER BY Battery_ID;