import os
import sqlite3
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

load_dotenv()
client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY"))
model = SentenceTransformer('all-MiniLM-L6-v2')

# This list will hold our memory
chat_history = []

def get_context(query):
    """Retrieves news based on the user's latest query."""
    embeddings = np.load('news_embeddings.npy')
    query_vec = model.encode(query)
    
    cos_scores = util.cos_sim(query_vec, embeddings)[0]
    top_indices = np.argsort(-cos_scores)[:3]
    
    conn = sqlite3.connect('news_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT title, description FROM articles")
    rows = cursor.fetchall()
    conn.close()
    
    context = ""
    for idx in top_indices:
        title, desc = rows[idx]
        context += f"\n- {title}: {desc}"
    return context

def chat():
    print("🤖 AI News Assistant is Online! (Type 'quit' to exit)")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
            
        # 1. Get relevant news context
        news_context = get_context(user_input)
        
        # 2. Build the message for the AI
        # We include the news context in the very first "System" message
        system_prompt = f"You are a helpful news assistant. Use this info: {news_context}"
        
        # 3. Prepare the payload (System prompt + History + New Question)
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(chat_history) # Add previous messages
        messages.append({"role": "user", "content": user_input}) # Add current question
        
        # 4. Get Response from Groq
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages
        )
        
        ai_response = response.choices[0].message.content
        print(f"\nAI: {ai_response}")
        
        # 5. Update Memory (Keep only last 5 exchanges to save space)
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": ai_response})
        
        if len(chat_history) > 10: # 5 user + 5 assistant messages
            chat_history.pop(0)
            chat_history.pop(0)

if __name__ == "__main__":
    chat()