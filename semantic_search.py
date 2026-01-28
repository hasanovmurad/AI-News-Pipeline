import sqlite3
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

def search_news(query):
    # 1. Load the Model and the Embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = np.load('news_embeddings.npy')
    
    # 2. Convert the user's query into a vector
    query_embedding = model.encode(query)

    # 3. Compute Similarity between the query and all stored news
    # This returns a list of scores
    cos_scores = util.cos_sim(query_embedding, embeddings)[0]
    
    # 4. Find the top 3 results
    top_results = torch.topk(cos_scores, k=3)
    
    # 5. Connect to SQL to get the actual titles and URLs
    conn = sqlite3.connect('news_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT title, url FROM articles")
    all_articles = cursor.fetchall()
    
    print(f"\n--- Top Results for: '{query}' ---")
    for score, idx in zip(top_results.values, top_results.indices):
        title, url = all_articles[idx]
        print(f"\nScore: {score:.4f}")
        print(f"Title: {title}")
        print(f"Link: {url}")
        
    conn.close()

if __name__ == "__main__":
    user_query = input("Enter a topic or question to search: ")
    search_news(user_query)