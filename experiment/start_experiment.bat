@echo off
cd C:\Users\paad\rbc-parameter-tuning\experiment_codes 
:start
python rbc_3_1_2.py
echo Programm closed due to an ERROR
timeout /t 5
GOTO:start

pause