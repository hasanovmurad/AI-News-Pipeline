import os
import sqlite3
import logging
import requests
import pandas as pd
from dotenv import load_dotenv

# 1. SETUP: Load environment variables and configure logging
load_dotenv()
logging.basicConfig(
    filename='pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 2. CONFIGURATION: Get API Key from .env
API_KEY = os.getenv('NEWS_API_KEY')
DB_NAME = 'news_data.db'

def save_to_database(articles):
    """Saves cleaned articles to SQLite with duplicate prevention."""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Ensure the table exists with a UNIQUE constraint on the title
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT UNIQUE, 
                source TEXT,
                date TEXT,
                url TEXT
            )
        ''')
        
        saved_count = 0
        skipped_count = 0

        for art in articles:
            try:
                cursor.execute('''
                    INSERT INTO articles (title, source, date, url)
                    VALUES (?, ?, ?, ?)
                ''', (art['title'], art['source'], art['date'], art['url']))
                saved_count += 1
            except sqlite3.IntegrityError:
                # Article already exists in DB
                skipped_count += 1
                continue
        
        conn.commit()
        conn.close()
        logging.info(f"Database sync complete. New: {saved_count}, Duplicates skipped: {skipped_count}")
        print(f"--- Database updated: {saved_count} new articles added ---")
        
    except Exception as e:
        logging.error(f"Database error: {e}")
        print(f"Critical Database Error: {e}")

def fetch_news(topic):
    """Fetches news from API using secure API key."""
    if not API_KEY:
        logging.error("API Key not found! Please check your .env file.")
        return []

    url = f'https://newsapi.org/v2/everything?q={topic}&apiKey={API_KEY}'
    logging.info(f"Initiating fetch for topic: {topic}")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        articles = data.get('articles', [])
        
        logging.info(f"Successfully retrieved {len(articles)} articles from API.")
        
        cleaned_list = []
        for art in articles:
            cleaned_list.append({
                'title': art.get('title'),
                'source': art.get('source', {}).get('name'),
                'date': art.get('publishedAt'),
                'url': art.get('url')
            })
        return cleaned_list
    except Exception as e:
        logging.error(f"API Request failed: {e}")
        return []

if __name__ == "__main__":
    logging.info("===== PIPELINE START =====")
    SEARCH_QUERY = 'artificial intelligence'
    
    # Execution Flow
    news_data = fetch_news(SEARCH_QUERY)
    
    if news_data:
        # Save to SQL
        save_to_database(news_data)
        
        # Optional: Save a quick CSV backup
        try:
            df = pd.DataFrame(news_data)
            df.to_csv('latest_news_backup.csv', index=False)
            logging.info("CSV backup generated.")
        except Exception as e:
            logging.error(f"CSV Export error: {e}")
            
    logging.info("===== PIPELINE END =====")
    print("Process complete. Check pipeline.log for details.")