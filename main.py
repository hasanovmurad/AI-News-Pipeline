import requests
import pandas as pd

def fetch_and_save_news(topic, api_key):
    url = f'https://newsapi.org/v2/everything?q={topic}&apiKey={api_key}'
    
    try:
        print(f"--- {topic} haberleri indiriliyor... ---")
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        articles = data.get('articles', [])
        cleaned_data = []
        
        for art in articles:
            cleaned_data.append({
                'title': art.get('title'),
                'source': art.get('source', {}).get('name'),
                'date': art.get('publishedAt'),
                'url': art.get('url') # Dünkü ödevin!
            })
            
        df = pd.DataFrame(cleaned_data)
        filename = f"{topic.replace(' ', '_')}_news.csv"
        df.to_csv(filename, index=False)
        print(f"Başarılı! {len(df)} haber {filename} dosyasına kaydedildi.")
        
    except Exception as e:
        print(f"Hata: {e}")

if __name__ == "__main__":
    # Kendi API anahtarını buraya yaz
    MY_KEY = 'fd6576ffc0a14ae0a3481f9de0997474'
    fetch_and_save_news('artificial intelligence', MY_KEY)