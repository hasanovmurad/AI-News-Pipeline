import streamlit as st
import os
import sqlite3
import numpy as np
import subprocess
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util, CrossEncoder

# --- 1. INITIAL SETUP & CONFIG ---
load_dotenv()
st.set_page_config(page_title="AI News Intelligence", page_icon="🤖", layout="wide")

# Initialize the Groq AI Client
client = OpenAI(
    base_url="https://api.groq.com/openai/v1", 
    api_key=os.getenv("GROQ_API_KEY")
)

# --- 2. CACHED MODELS (Loads once, stays in memory) ---
@st.cache_resource
def load_ai_models():
    """Load the Embedding model and the Reranker model."""
    bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return bi_encoder, cross_encoder

embedding_model, reranker_model = load_ai_models()

# --- 3. CORE LOGIC FUNCTIONS ---

def generate_multi_queries(original_query):
    """Rewrite the user query into 3 versions to improve search accuracy."""
    prompt = f"""You are an AI Search Optimizer. 
    Rewrite the following user question into 3 different versions to find better news results.
    Output only the questions, one per line.
    ORIGINAL QUESTION: {original_query}"""
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    
    queries = response.choices[0].message.content.strip().split("\n")
    return [q.strip() for q in queries if q.strip()] + [original_query]

def get_context(query):
    """The RAG Pipeline: Search -> Retrieve -> Rerank."""
    if not os.path.exists('news_embeddings.npy'):
        return "No database found. Please run the data scraper first."
    
    # A. Multi-Query Search
    all_search_terms = generate_multi_queries(query)
    embeddings = np.load('news_embeddings.npy')
    
    # Connect to Database
    conn = sqlite3.connect('news_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT title, description, url FROM articles")
    all_articles = cursor.fetchall()
    conn.close()

    # B. Vector Search (Bi-Encoder)
    candidate_indices = set()
    for term in all_search_terms:
        query_vec = embedding_model.encode(term)
        scores = util.cos_sim(query_vec, embeddings)[0]
        top_hits = np.argsort(-scores)[:3] # Get top 3 for each query version
        for idx in top_hits:
            candidate_indices.add(int(idx))

    # C. Deep Ranking (Cross-Encoder)
    candidates = [all_articles[idx] for idx in candidate_indices]
    # Pair the query with each candidate: [Question, Article Content]
    ranking_pairs = [[query, f"{c[0]} {c[1]}"] for c in candidates]
    rerank_scores = reranker_model.predict(ranking_pairs)
    
    # Sort articles by the new high-quality scores
    ranked_data = sorted(zip(rerank_scores, candidates), key=lambda x: x[0], reverse=True)

    # D. Format Final Context (Top 3 results)
    context_text = ""
    for score, article in ranked_data[:3]:
        title, desc, url = article
        context_text += f"\n- **{title}**: {desc} ([Read More]({url}))"
    return context_text

# --- 4. SIDEBAR DASHBOARD ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.title("Admin Dashboard")
    
    # Show Database Stats
    if os.path.exists('news_data.db'):
        conn = sqlite3.connect('news_data.db')
        count = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        conn.close()
        st.metric("Articles in DB", count)
    else:
        st.error("Database not found!")

    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.info("System: RAG Pipeline v2.0\nModels: Llama 3.1 + MiniLM")
    st.divider()
    st.header("🔄 Data Management")

    if st.button("Fetch & Embed Latest News"):
        with st.status("Running Pipeline...", expanded=True) as status:
            st.write("1. Scraping latest AI news...")
            # Runs your main.py script
            subprocess.run(["python", "main.py"])
            
            st.write("2. Generating new vector embeddings...")
            # Runs your embed_news.py script
            subprocess.run(["python", "embed_news.py"])
            
            status.update(label="Pipeline Complete!", state="complete", expanded=False)
        
        st.success("Database is now up to date!")
        st.rerun() # Refresh the app to show new stats

# --- 5. MAIN CHAT INTERFACE ---
st.title("🌐 AI News Intelligence")
st.caption("Advanced RAG System with Multi-Query Expansion & Reranking")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if user_input := st.chat_input("Ask about AI trends, jobs, or news..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing and Verifying..."):
            # 1. Fetch context
            context = get_context(user_input)
            
            # 2. Generate the primary answer
            system_instruction = f"You are a professional News Analyst. Use this data: {context}."
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "system", "content": system_instruction}] + st.session_state.messages
            )
            answer = response.choices[0].message.content

            # 3. NEW: Self-Evaluation Logic (The Groundedness Check)
            eval_prompt = f"""
            Compare the following ANSWER to the provided CONTEXT. 
            Rate the 'Groundedness' (how much the answer stays true to the context) on a scale of 1 to 10.
            1 = Made up entirely. 10 = Fully supported by the news.
            Output ONLY the number.
            
            CONTEXT: {context}
            ANSWER: {answer}
            """
            eval_response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": eval_prompt}]
            )
            confidence_score = eval_response.choices[0].message.content.strip()

            # 4. Display results with a "Trust Meter"
            st.markdown(answer)
            
            # Show a colored status based on the score
            score_num = int(confidence_score) if confidence_score.isdigit() else 5
            if score_num >= 8:
                st.success(f"Confidence Score: {score_num}/10 (High Reliability)")
            elif score_num >= 5:
                st.warning(f"Confidence Score: {score_num}/10 (Moderate Reliability)")
            else:
                st.error(f"Confidence Score: {score_num}/10 (Low Reliability - Possible Hallucination)")

            with st.expander("📚 View Sources (Reranked)"):
                st.write(context)

    st.session_state.messages.append({"role": "assistant", "content": answer})