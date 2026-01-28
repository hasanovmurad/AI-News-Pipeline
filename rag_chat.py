import os
import sqlite3
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI  # Groq uses the OpenAI library format
from sentence_transformers import SentenceTransformer, util

load_dotenv()

# 1. Setup the "Brain" (LLM)
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

def get_context(query):
    """Retrieves the most relevant news from your SQL/NPY database."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = np.load('news_embeddings.npy')
    query_embedding = model.encode(query)
    
    # Get top 3 results
    cos_scores = util.cos_sim(query_embedding, embeddings)[0]
    top_results = np.argsort(-cos_scores)[:3] # Getting indices of top 3
    
    conn = sqlite3.connect('news_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT title FROM articles")
    all_titles = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    context_text = "\n".join([all_titles[idx] for idx in top_results])
    return context_text

def ask_ai(question):
    # Step A: Get relevant news from our database
    context = get_context(question)
    
    # Step B: Create the prompt
    prompt = f"""
    You are a helpful AI News Assistant. 
    Use the following news headlines to answer the user's question.
    If the answer isn't in the headlines, say you don't know based on the current data.
    
    HEADLINES:
    {context}
    
    USER QUESTION: 
    {question}
    """

    # Step C: Generate the answer using an ACTIVE model
    # Switching to llama-3.1-8b-instant (the direct successor)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant", 
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

if __name__ == "__main__":
    user_input = input("Ask me anything about your AI news: ")
    answer = ask_ai(user_input)
    print("\n--- AI ANSWER ---")
    print(answer)