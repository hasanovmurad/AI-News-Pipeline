# AI News Data Pipeline

## 📌 Overview
This is the first project of my AI Portfolio. It is a Python-based data pipeline that fetches live news about Artificial Intelligence using the NewsAPI, cleans the data, and exports it as a structured CSV file.

## 🛠️ Tech Stack
- **Language:** Python 3.x
- **Libraries:** Requests (API handling), Pandas (Data manipulation)
- **Version Control:** Git & GitHub
- **Format:** JSON to CSV

## 🚀 Features
- Connects to a REST API to retrieve real-time data.
- Implements error handling for network requests.
- Processes nested JSON data into a clean tabular format.
- Adds metadata like publication date, source, and URL for each article.

## 📂 Project Structure
- `main.py`: The core script that executes the pipeline.
- `*_news.csv`: Automated output files generated based on the search topic.
- `venv/`: Virtual environment to keep dependencies isolated.

## 📈 Learning Journey
In this first week, I mastered:
1. Setting up a professional development environment.
2. Handling API authentication and requests.
3. Data cleaning and structural transformation with Pandas.
4. Version control best practices with Git.

## 🤖 Automation
The pipeline is fully automated using a Windows Batch script (`run_pipeline.bat`) and Windows Task Scheduler.
- **Schedule:** Runs daily at 08:00 AM.
- **Process:** Activates `venv`, fetches news, updates the SQLite database, and regenerates the visualization chart.
- **Reliability:** Uses a logging system (to be implemented) to track background execution.