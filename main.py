import requests
import pandas as pd

def fetch_and_save_news(topic, api_key):
    # API Endpoint construction
    url = f'https://newsapi.org/v2/everything?q={topic}&apiKey={api_key}'
    
    try:
        print(f"--- Fetching news for: {topic} ---")
        response = requests.get(url)
        response.raise_for_status() # Check for HTTP errors
        data = response.json()
        
        articles = data.get('articles', [])
        cleaned_data = []
        
        # Iterating through results and extracting relevant fields
        for art in articles:
            cleaned_data.append({
                'title': art.get('title'),
                'source': art.get('source', {}).get('name'),
                'date': art.get('publishedAt'),
                'url': art.get('url')
            })
            
        # Converting list of dicts to a Pandas DataFrame
        df = pd.DataFrame(cleaned_data)
        filename = f"{topic.replace(' ', '_')}_news.csv"
        
        # Exporting to CSV
        df.to_csv(filename, index=False)
        print(f"Success! {len(df)} articles saved to {filename}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    MY_KEY = 'fd6576ffc0a14ae0a3481f9de0997474' # Replace with your actual NewsAPI key
    fetch_and_save_news('artificial intelligence', MY_KEY)