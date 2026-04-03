USE [BATTERY_DB]; -- Βεβαιώσου ότι το όνομα είναι σωστό
GO

-- Δημιουργία του πίνακα χαρακτηριστικών
CREATE TABLE CYCLE_FEATURES (
    Feature_ID INT IDENTITY(1,1) PRIMARY KEY, -- Αυτόματος αύξων αριθμός
    Battery_ID VARCHAR(50),
    Cycle_Index INT,
    Discharge_Time_Seconds FLOAT,
    Avg_Temperature FLOAT,
    Max_Temperature FLOAT,
    Capacity_Ah FLOAT
);
GO