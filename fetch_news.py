import requests

# 1. Hazırlık
API_KEY = 'fd6576ffc0a14ae0a3481f9de0997474'  # NewsAPI anahtarınız
KONU = 'artificial intelligence' # AI ile ilgili haberleri arayalım
URL = f'https://newsapi.org/v2/everything'

print(f"--- {KONU} konusu için haberler getiriliyor... ---")

# 2. İnternet üzerinden veriyi iste (Request)
params = {
    'q': KONU,
    'apiKey': API_KEY,
    'sortBy': 'publishedAt'
}
response = requests.get(URL, params=params, timeout=20)

# 3. Gelen veriyi Python'un anlayacağı dile çevir (JSON)
data = response.json()  

## Veriyi temizle ve düzenle

# 1. BOŞ BİR LİSTE OLUŞTUR (Temiz veriler buraya gelecek)
cleaned_news = []

if data['status'] == 'ok':
    articles = data['articles']
    
    for art in articles:
        # 2. SADECE İSTEDİĞİMİZ BİLGİLERİ SEÇİYORUZ
        temp_dict = {
            'baslik': art.get('title'),
            'ozet': art.get('description'),
            'tarih': art.get('publishedAt'),
            'kaynak': art.get('source', {}).get('name')
        }
        
        # 3. LİSTEYE EKLE
        cleaned_news.append(temp_dict)

# 4. SONUCU GÖR (Sadece ilk 2 temiz haberi yazdıralım)
print(f"Toplam {len(cleaned_news)} adet haber temizlendi.")
print("Örnek Veri:", cleaned_news[3])