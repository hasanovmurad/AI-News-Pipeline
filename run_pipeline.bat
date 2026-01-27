@echo off
:: Change the directory to your project folder
cd /d "D:\User\Desktop\AI_Portfolio_Project"

:: Activate the virtual environment
call venv\Scripts\activate

:: Run the main pipeline
python main.py

:: (Optional) Update the chart as well
python visualize_data.py

echo Pipeline executed successfully at %date% %time%
pause