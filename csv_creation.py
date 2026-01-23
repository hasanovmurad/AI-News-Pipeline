import requests
import pandas as pd # 'pd' kısaltması standarttır

API_KEY = 'fd6576ffc0a14ae0a3481f9de0997474' # NewsAPI anahtarınız
URL = f'https://newsapi.org/v2/everything?q=artificial intelligence&apiKey={API_KEY}' 

# 1. Veriyi çek
response = requests.get(URL)
data = response.json()
articles = data.get('articles', [])

# 2. Veriyi listeye hazırla (Dünkü temizlik)
news_list = []
for art in articles:
    news_list.append({
        'title': art.get('title'),
        'source': art.get('source', {}).get('name'),
        'date': art.get('publishedAt'),
        'url': art.get('url')
    })

# 3. PANDAS SİHRİ: Listeyi Tabloya (DataFrame) Çevir
df = pd.DataFrame(news_list)

# 4. Tabloyu ekranda gör (İlk 5 satır)
print("Tablonun İlk Hali:")
print(df.head())

# 5. Bilgisayara Kaydet (CSV Formatında)
df.to_csv('AI_News.csv', index=False, encoding='utf-8')
print("\nVeriler 'AI_News.csv' olarak kaydedildi!")