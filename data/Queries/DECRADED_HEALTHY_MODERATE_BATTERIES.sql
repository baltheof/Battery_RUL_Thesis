SELECT Battery_ID,
       AVG(Health_actual) AS Health_end,
       CASE WHEN AVG(Health_actual) > 0.9 THEN 'Healthy'
            WHEN AVG(Health_actual) > 0.8 THEN 'Moderate'
            ELSE 'Degraded' END AS Class_end
FROM (
    SELECT Battery_ID, Health_actual,
           ROW_NUMBER() OVER (PARTITION BY Battery_ID ORDER BY Cycle_Index DESC) AS rn
    FROM HEALTH_PRED_MLP
) t
WHERE rn <= 5
GROUP BY Battery_ID
ORDER BY Battery_ID;