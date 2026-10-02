SELECT Battery_ID,
       MAX(Is_Censored) AS Is_Censored,
       COUNT(*) AS Cycles,
       MAX(RUL) AS RUL_max
FROM CYCLE_FEATURES_CENSORED
GROUP BY Battery_ID
ORDER BY Battery_ID