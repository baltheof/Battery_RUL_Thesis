SELECT 
    Battery_ID,
    COUNT(*) AS Total_Cycles,
    ROUND(MIN(SoH), 3) AS Min_SoH,
    ROUND(MAX(SoH), 3) AS Max_SoH,
    ROUND(AVG(SoH), 3) AS Avg_SoH,
    ROUND(MAX(Nominal), 3) AS Nominal,
    MIN(RUL) AS RUL_min,
    MAX(RUL) AS RUL_max
FROM CYCLE_FEATURES
GROUP BY Battery_ID
ORDER BY Battery_ID