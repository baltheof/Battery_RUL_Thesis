USE [BATTERY_DB]; 
GO

SELECT 
    C.Battery_ID,             
    C.Cycle_Index,            
    C.Capacity_Ah,            
    M.Time_Seconds,           
    M.Voltage_Measured,       
    M.Current_Measured,       
    M.Temperature_Measure     
FROM MEASUREMENTS M
INNER JOIN TEST_CYCLES C ON M.Cycle_ID = C.Cycle_ID
WHERE
    C.Operation_Type = 'discharge'  
ORDER BY 
    C.Battery_ID,
    C.Cycle_Index, 
    M.Time_Seconds;