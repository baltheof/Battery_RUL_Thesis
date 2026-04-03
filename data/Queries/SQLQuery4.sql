INSERT INTO CYCLE_FEATURES (Battery_ID, Cycle_Index, Discharge_Time_Seconds, Avg_Temperature, Max_Temperature, Capacity_Ah)
SELECT 
    C.Battery_ID,
    C.Cycle_Index,
    MAX(M.Time_Seconds) - MIN(M.Time_Seconds) AS Discharge_Time, -- Υπολογισμός χρόνου
    AVG(M.Temperature_Measure) AS Avg_Temp,                    -- Μέση Θερμοκρασία
    MAX(M.Temperature_Measure) AS Max_Temp,                    -- Μέγιστη Θερμοκρασία
    MAX(C.Capacity_Ah) AS Capacity                             -- Η χωρητικότητα του κύκλου
FROM MEASUREMENTS M
JOIN TEST_CYCLES C ON M.Cycle_ID = C.Cycle_ID
WHERE C.Battery_ID = 'B0047' AND C.Operation_Type = 'discharge'
GROUP BY C.Battery_ID, C.Cycle_Index;