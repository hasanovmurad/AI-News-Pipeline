import requests
import sqlite3
import pandas as pd

def save_to_database(articles):
    """Saves a list of articles to the SQLite database."""
    try:
        conn = sqlite3.connect('news_data.db')
        cursor = conn.cursor()
        
        # Ensure the table exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT UNIQUE, 
                source TEXT,
                date TEXT,
                url TEXT
            )
        ''')
        
        # Insert articles
        for art in articles:
            try:
                cursor.execute('''
                    INSERT INTO articles (title, source, date, url)
                    VALUES (?, ?, ?, ?)
                ''', (art['title'], art['source'], art['date'], art['url']))
            except sqlite3.IntegrityError:
                # This skips the article if the title already exists (Unique constraint)
                continue
        
        conn.commit()
        conn.close()
        print("--- Database Update: Completed ---")
    except Exception as e:
        print(f"Database Error: {e}")

def fetch_news(topic, api_key):
    """Fetches news from API and returns a list of cleaned dictionaries."""
    url = f'https://newsapi.org/v2/everything?q={topic}&apiKey={api_key}'
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        articles = data.get('articles', [])
        
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
        print(f"API Error: {e}")
        return []

if __name__ == "__main__":
    API_KEY = 'fd6576ffc0a14ae0a3481f9de0997474' # Example API Key
    QUERY = 'artificial intelligence'
    
    # Step 1: Fetch
    news_data = fetch_news(QUERY, API_KEY)
    
    if news_data:
        # Step 2: Save to SQL
        save_to_database(news_data)
        
        # Step 3: Save to CSV (for backup)
        df = pd.DataFrame(news_data)
        df.to_csv('latest_news_backup.csv', index=False)
        print(f"--- Process Finished: {len(news_data)} articles processed ---")
    