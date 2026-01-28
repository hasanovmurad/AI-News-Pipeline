import sqlite3
import numpy as np # Library for handling large lists of numbers
from sentence_transformers import SentenceTransformer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def generate_database_embeddings():
    # 1. Load the AI Model (The same one from yesterday)
    logging.info("Loading AI model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 2. Connect to SQL and fetch titles
    conn = sqlite3.connect('news_data.db')
    cursor = conn.cursor()
    
    logging.info("Fetching titles from SQL...")
    cursor.execute("SELECT title FROM articles")
    rows = cursor.fetchall()
    
    # Flatten the list of tuples into a simple list of strings
    titles = [row[0] for row in rows if row[0] is not None]
    
    if not titles:
        logging.warning("No titles found in the database. Run your main.py first!")
        return

    # 3. Convert ALL titles to math (Embeddings)
    logging.info(f"Turning {len(titles)} titles into vectors. Please wait...")
    embeddings = model.encode(titles, show_progress_bar=True)

    # 4. Save the results
    # We save these as a .npy file (a format for fast math loading)
    np.save('news_embeddings.npy', embeddings)
    
    logging.info("--- Success! ---")
    logging.info(f"Generated a matrix of shape: {embeddings.shape}")
    logging.info("Vectors saved to 'news_embeddings.npy'")
    
    conn.close()

if __name__ == "__main__":
    generate_database_embeddings()