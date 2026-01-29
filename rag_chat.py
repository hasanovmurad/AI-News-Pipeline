import os
import sqlite3
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

load_dotenv()

# 1. Setup the LLM Client (Groq)
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

def get_context(query):
    """Retrieves titles AND descriptions for better AI context."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load the fresh embeddings you created in the previous step
    if not os.path.exists('news_embeddings.npy'):
        return "Error: Embeddings file not found. Please run embed_news.py first."
        
    embeddings = np.load('news_embeddings.npy')
    query_embedding = model.encode(query)
    
    # Get top 3 most relevant indices
    cos_scores = util.cos_sim(query_embedding, embeddings)[0]
    top_results = np.argsort(-cos_scores)[:3]
    
    # 2. Connect to SQL and fetch BOTH Title and Description
    conn = sqlite3.connect('news_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT title, description FROM articles")
    rows = cursor.fetchall()
    conn.close()
    
    # 3. Format the context strings
    context_list = []
    for idx in top_results:
        title, desc = rows[idx]
        # We combine them so the AI sees the whole story
        context_list.append(f"TITLE: {title}\nDETAILS: {desc}")
    
    return "\n\n---\n\n".join(context_list)

def ask_ai(question):
    # Retrieve the enhanced context
    context = get_context(question)
    
    # 4. Refined Prompt (More professional)
    prompt = f"""
    You are a professional AI News Analyst. 
    Use the following news articles to provide a detailed and insightful answer to the user.
    If the context doesn't contain enough info, state that clearly.

    ARTICLES FOUND IN DATABASE:
    {context}
    
    USER QUESTION: 
    {question}
    
    YOUR ANALYSIS:
    """

    # Using the active llama-3.1-8b-instant model
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant", 
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

if __name__ == "__main__":
    user_input = input("Enter your AI news question: ")
    print("\nSearching database and thinking...")
    answer = ask_ai(user_input)
    print("\n--- AI ANALYSIS ---")
    print(answer)