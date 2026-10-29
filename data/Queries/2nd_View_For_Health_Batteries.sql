USE BATTERY_DB;
GO

CREATE OR ALTER VIEW PRED_LINEAR_SUMMARY AS
SELECT Battery_ID,
       COUNT(*) AS Cycles,
       ROUND(AVG(ABS(Pred_linear - Health_actual)), 3)   AS MAE_Model,
       ROUND(AVG(ABS(Pred_baseline - Health_actual)), 3) AS MAE_Baseline,
       ROUND(AVG(Pred_linear - Health_actual), 3)        AS Bias_Model,
       ROUND(MAX(Health_actual), 3) AS Actual_Max,
       ROUND(MIN(Health_actual), 3) AS Actual_Min,
       ROUND(MAX(Pred_linear), 3)   AS Model_Max,
       ROUND(MIN(Pred_linear), 3)   AS Model_Min,
       CASE WHEN AVG(ABS(Pred_linear - Health_actual))
               < AVG(ABS(Pred_baseline - Health_actual))
            THEN 'yes' ELSE 'no' END AS Model_Wins
FROM HEALTH_PRED_LINEAR
GROUP BY Battery_ID;
GO