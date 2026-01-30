"""
Enhanced Multimodal News Agent
A production-ready AI news assistant with vision, voice, and RAG capabilities
"""

import streamlit as st
import os
import sqlite3
import numpy as np
import io
import base64
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from contextlib import contextmanager
import logging

import tools
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from streamlit_mic_recorder import mic_recorder

# --- CONFIGURATION ---
class Config:
    """Centralized configuration management"""
    TEXT_MODEL = "llama-3.3-70b-versatile"
    VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
    WHISPER_MODEL = "whisper-large-v3"
    
    DB_PATH = Path("news_data.db")
    EMBEDDINGS_PATH = Path("news_embeddings.npy")
    
    MAX_CONTEXT_ARTICLES = 3
    RETRIEVAL_CANDIDATES = 10
    PDF_CONTEXT_LENGTH = 2000
    MIN_AUDIO_BYTES = 2000
    
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
    RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- INITIALIZATION ---
load_dotenv()

st.set_page_config(
    page_title="🤖 Multimodal News Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .source-link {
        background-color: #f0f2f6;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 0.25rem 0;
        border-left: 3px solid #667eea;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- CLIENT INITIALIZATION ---
@st.cache_resource
def get_openai_client() -> OpenAI:
    """Initialize and cache OpenAI client"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    
    return OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key
    )

@st.cache_resource
def load_ai_models() -> Tuple[SentenceTransformer, CrossEncoder]:
    """Load and cache AI models for embeddings and reranking"""
    try:
        logger.info("Loading AI models...")
        bi_encoder = SentenceTransformer(Config.EMBEDDING_MODEL)
        cross_encoder = CrossEncoder(Config.RERANKER_MODEL)
        logger.info("AI models loaded successfully")
        return bi_encoder, cross_encoder
    except Exception as e:
        logger.error(f"Failed to load AI models: {e}")
        st.error("Failed to initialize AI models. Please check your installation.")
        raise

# Initialize clients and models
try:
    client = get_openai_client()
    embedding_model, reranker_model = load_ai_models()
except Exception as e:
    st.error(f"Initialization failed: {e}")
    st.stop()

# --- DATABASE MANAGEMENT ---
@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = None
    try:
        conn = sqlite3.connect(Config.DB_PATH)
        yield conn
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            conn.close()

def initialize_database():
    """Create database tables if they don't exist"""
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

def save_message(role: str, content: str) -> bool:
    """Save a message to chat history"""
    try:
        with get_db_connection() as conn:
            conn.execute(
                "INSERT INTO chat_history (role, content) VALUES (?, ?)",
                (role, str(content))
            )
            conn.commit()
        return True
    except Exception as e:
        logger.error(f"Failed to save message: {e}")
        return False

def get_chat_history(limit: int = 50) -> List[Dict]:
    """Retrieve recent chat history"""
    try:
        with get_db_connection() as conn:
            cursor = conn.execute(
                "SELECT role, content, created_at FROM chat_history ORDER BY id DESC LIMIT ?",
                (limit,)
            )
            rows = cursor.fetchall()
            return [
                {"role": row[0], "content": row[1], "timestamp": row[2]}
                for row in reversed(rows)
            ]
    except Exception as e:
        logger.error(f"Failed to retrieve chat history: {e}")
        return []

def clear_chat_history() -> bool:
    """Clear all chat history"""
    try:
        with get_db_connection() as conn:
            conn.execute("DELETE FROM chat_history")
            conn.commit()
        return True
    except Exception as e:
        logger.error(f"Failed to clear chat history: {e}")
        return False

# --- RAG FUNCTIONS ---
def get_context(query: str) -> Tuple[str, List[Dict]]:
    """
    Retrieve and rerank relevant articles
    Returns: (formatted_context_string, list_of_source_dicts)
    """
    if not Config.EMBEDDINGS_PATH.exists():
        return "📭 Database is empty. Please add articles first.", []
    
    try:
        embeddings = np.load(Config.EMBEDDINGS_PATH)
        
        with get_db_connection() as conn:
            cursor = conn.execute("SELECT title, description, url FROM articles")
            all_articles = cursor.fetchall()
        
        if not all_articles:
            return "📭 No articles found in database.", []
        
        # Semantic search
        query_vec = embedding_model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_vec, embeddings)[0]
        
        # Get top candidates
        top_indices = np.argsort(-scores.cpu().numpy())[:Config.RETRIEVAL_CANDIDATES]
        candidates = [all_articles[i] for i in top_indices]
        
        # Rerank with cross-encoder
        pairs = [[query, f"{c[0]} {c[1]}"] for c in candidates]
        rerank_scores = reranker_model.predict(pairs)
        
        # Sort by reranked scores
        ranked = sorted(
            zip(rerank_scores, candidates),
            key=lambda x: x[0],
            reverse=True
        )[:Config.MAX_CONTEXT_ARTICLES]
        
        # Format context
        context_parts = []
        sources = []
        
        for score, (title, description, url) in ranked:
            context_parts.append(f"**{title}**\n{description}\nSource: {url}")
            sources.append({
                "title": title,
                "description": description,
                "url": url,
                "relevance_score": float(score)
            })
        
        formatted_context = "\n\n".join(context_parts)
        return formatted_context, sources
        
    except Exception as e:
        logger.error(f"Error in get_context: {e}")
        return f"⚠️ Error retrieving context: {str(e)}", []

# --- TOOL DEFINITIONS ---
tools_list = [
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": "Search the local database for AI-related news articles using semantic search. Use this when the user asks about news, articles, or specific topics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant articles"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_db_stats",
            "description": "Get statistics about the article database, including total count and recent additions.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information not in the local database. Use this for real-time news or topics not covered locally.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The web search query"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# --- SESSION STATE INITIALIZATION ---
def initialize_session_state():
    """Initialize all session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "sources" not in st.session_state:
        st.session_state.sources = []
    
    if "pdf_text" not in st.session_state:
        st.session_state.pdf_text = None
    
    if "visual_context" not in st.session_state:
        st.session_state.visual_context = None
    
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")

initialize_session_state()
initialize_database()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 🎛️ Control Panel")
    
    # Stats
    with st.expander("📊 Database Stats", expanded=True):
        try:
            stats = tools.get_db_stats()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Articles", stats.get("total_articles", 0))
            with col2:
                st.metric("Messages", len(st.session_state.messages))
        except Exception as e:
            st.error(f"Error loading stats: {e}")
    
    st.markdown("---")
    
    # Media uploads
    st.markdown("### 📂 Media Upload")
    
    uploaded_image = st.file_uploader(
        "📸 Upload Image",
        type=["png", "jpg", "jpeg", "webp"],
        help="Upload an image for visual analysis"
    )
    
    if uploaded_image:
        st.image(uploaded_image, use_container_width=True)
        if st.button("🔄 Analyze Image"):
            st.session_state.visual_context = None
    
    uploaded_pdf = st.file_uploader(
        "📄 Upload PDF",
        type="pdf",
        help="Upload a PDF document for context"
    )
    
    if uploaded_pdf and st.session_state.pdf_text is None:
        with st.spinner("📖 Reading PDF..."):
            try:
                pdf_content = tools.read_pdf(uploaded_pdf.read())
                st.session_state.pdf_text = pdf_content
                st.success(f"✅ PDF indexed ({len(pdf_content)} characters)")
            except Exception as e:
                st.error(f"PDF processing failed: {e}")
    
    st.markdown("---")
    
    # Controls
    st.markdown("### ⚙️ Actions")
    
    if st.button("🗑️ Clear Chat History", type="secondary"):
        if clear_chat_history():
            st.session_state.messages = []
            st.session_state.sources = []
            st.success("Chat cleared!")
            st.rerun()
    
    if st.button("🔄 Reset Session", type="secondary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # Download chat
    if st.session_state.messages:
        chat_export = json.dumps(st.session_state.messages, indent=2)
        st.download_button(
            "💾 Download Chat",
            data=chat_export,
            file_name=f"chat_{st.session_state.conversation_id}.json",
            mime="application/json"
        )

# --- MAIN INTERFACE ---
st.markdown('<h1 class="main-header">🤖 Multimodal News Agent</h1>', unsafe_allow_html=True)
st.markdown("Ask me about AI news, upload documents, or use voice input!")

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] != "system" and msg.get("role") != "tool":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# Display sources if available
if st.session_state.sources:
    with st.expander("📚 Sources Used", expanded=False):
        for i, source in enumerate(st.session_state.sources, 1):
            st.markdown(f"""
            <div class="source-link">
                <strong>{i}. {source['title']}</strong><br>
                {source['description']}<br>
                <a href="{source['url']}" target="_blank">🔗 Read more</a>
                <small>(Relevance: {source['relevance_score']:.2f})</small>
            </div>
            """, unsafe_allow_html=True)

# --- INPUT HANDLING ---
user_input = ""

# Voice input
col1, col2 = st.columns([3, 1])
with col2:
    speech = mic_recorder(
        start_prompt="🎤 Start Recording",
        stop_prompt="🛑 Stop",
        key='mic_recorder'
    )

if speech and speech.get('bytes') and len(speech['bytes']) > Config.MIN_AUDIO_BYTES:
    with st.spinner("🎧 Transcribing..."):
        try:
            transcription = client.audio.transcriptions.create(
                file=("audio.wav", io.BytesIO(speech['bytes'])),
                model=Config.WHISPER_MODEL,
                language="en"
            )
            user_input = transcription.text
            st.success(f"🎤 Transcribed: {user_input}")
        except Exception as e:
            st.error(f"Transcription failed: {e}")
            logger.error(f"Transcription error: {e}")

# Text input
with col1:
    text_input = st.chat_input("💬 Type your message here...")
    if text_input:
        user_input = text_input

# --- AGENT EXECUTION ---
if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    save_message("user", user_input)
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        # STEP 1: Vision Analysis (if image provided and not analyzed)
        visual_context = ""
        if uploaded_image and st.session_state.visual_context is None:
            with st.status("👁️ Analyzing image...", expanded=True) as status:
                try:
                    b64_image = base64.b64encode(uploaded_image.getvalue()).decode("utf-8")
                    
                    vision_response = client.chat.completions.create(
                        model=Config.VISION_MODEL,
                        messages=[{
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Describe this image in detail. Identify any companies, brands, products, people, or key subjects. Be specific and comprehensive."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{b64_image}"
                                    }
                                }
                            ]
                        }],
                        max_tokens=500
                    )
                    
                    visual_context = vision_response.choices[0].message.content
                    st.session_state.visual_context = visual_context
                    
                    st.markdown(f"**🔍 Visual Analysis:**\n{visual_context}")
                    status.update(label="✅ Image analyzed!", state="complete")
                    
                except Exception as e:
                    logger.error(f"Vision analysis failed: {e}")
                    st.error(f"Vision analysis failed: {e}")
                    status.update(label="❌ Vision analysis failed", state="error")
        
        elif st.session_state.visual_context:
            visual_context = st.session_state.visual_context
        
        # STEP 2: Agent Reasoning & Tool Use
        with st.status("🧠 Thinking...", expanded=True) as status:
            try:
                # Prepare system context
                pdf_context = ""
                if st.session_state.pdf_text:
                    pdf_context = f"\n\nPDF Document Context (first {Config.PDF_CONTEXT_LENGTH} chars):\n{st.session_state.pdf_text[:Config.PDF_CONTEXT_LENGTH]}"
                
                visual_info = ""
                if visual_context:
                    visual_info = f"\n\nVisual Context from uploaded image:\n{visual_context}"
                
                system_message = {
                    "role": "system",
                    "content": f"""You are an expert AI news assistant with access to multiple capabilities:

1. **Local News Database**: Search using the search_news tool for AI-related articles
2. **Web Search**: Use web_search tool for current information beyond the database
3. **Database Statistics**: Use get_db_stats to report on available articles
4. **Document Analysis**: Reference PDF content when available
5. **Visual Analysis**: Use image context when provided

Your goal is to provide accurate, helpful, and well-sourced responses. Always cite your sources and be transparent about where information comes from.

{pdf_context}{visual_info}

Be conversational, helpful, and comprehensive. When using tools, explain what you're doing."""
                }
                
                # Prepare messages for API call
                api_messages = [system_message] + [
                    msg for msg in st.session_state.messages
                    if msg.get("role") != "tool"
                ]
                
                # First API call with tools
                response = client.chat.completions.create(
                    model=Config.TEXT_MODEL,
                    messages=api_messages,
                    tools=tools_list,
                    tool_choice="auto",
                    temperature=0.7,
                    max_tokens=2000
                )
                
                response_message = response.choices[0].message
                tool_calls = response_message.tool_calls
                
                # Handle tool calls
                if tool_calls:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            }
                            for tc in tool_calls
                        ]
                    })
                    
                    for tool_call in tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)
                        
                        st.markdown(f"🔧 **Using tool:** `{function_name}`")
                        st.json(function_args)
                        
                        # Execute tool
                        tool_result = ""
                        try:
                            if function_name == "search_news":
                                query = function_args.get("query", user_input)
                                context, sources = get_context(query)
                                tool_result = context
                                st.session_state.sources = sources
                                
                            elif function_name == "get_db_stats":
                                tool_result = json.dumps(tools.get_db_stats(), indent=2)
                                
                            elif function_name == "web_search":
                                query = function_args.get("query", "")
                                tool_result = tools.web_search(query)
                            
                            st.success(f"✅ Tool executed successfully")
                            
                        except Exception as e:
                            tool_result = f"Error executing {function_name}: {str(e)}"
                            st.error(tool_result)
                            logger.error(f"Tool execution error: {e}")
                        
                        # Add tool result to messages
                        st.session_state.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": str(tool_result)
                        })
                    
                    # Second API call to get final response
                    final_response = client.chat.completions.create(
                        model=Config.TEXT_MODEL,
                        messages=[system_message] + st.session_state.messages,
                        temperature=0.7,
                        max_tokens=2000
                    )
                    
                    final_answer = final_response.choices[0].message.content
                else:
                    final_answer = response_message.content
                
                status.update(label="✅ Complete!", state="complete")
                
            except Exception as e:
                final_answer = f"❌ An error occurred: {str(e)}\n\nPlease try again or rephrase your question."
                logger.error(f"Agent execution error: {e}")
                status.update(label="❌ Error occurred", state="error")
        
        # Display final response
        response_placeholder.markdown(final_answer)
        
        # Save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_answer
        })
        save_message("assistant", final_answer)

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <small>Powered by Groq • Llama 3.3 70B • Sentence Transformers</small><br>
    <small>💡 Tip: Try voice input, upload images, or ask about AI news!</small>
</div>
""", unsafe_allow_html=True)