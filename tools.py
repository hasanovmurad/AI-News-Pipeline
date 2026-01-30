"""
Enhanced Tools Module for Multimodal News Agent - Day 37 Edition
Adds intelligent fallback logic and web search orchestration
"""

import sqlite3
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import requests
from PyPDF2 import PdfReader
import io
import os

logger = logging.getLogger(__name__)

# Configuration
DB_PATH = Path("news_data.db")
SERPER_API_KEY = None

# Minimum threshold for local search results
MIN_LOCAL_RESULTS = 2


def read_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes"""
    try:
        pdf_file = io.BytesIO(pdf_bytes)
        pdf_reader = PdfReader(pdf_file)
        
        text_content = []
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text_content.append(f"--- Page {page_num + 1} ---\n{page_text}")
            except Exception as e:
                logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                continue
        
        full_text = "\n\n".join(text_content)
        logger.info(f"Successfully extracted {len(full_text)} characters from PDF")
        
        return full_text if full_text else "No text could be extracted from PDF"
        
    except Exception as e:
        logger.error(f"PDF reading error: {e}")
        raise Exception(f"Failed to read PDF: {str(e)}")


def get_db_stats() -> Dict[str, Any]:
    """Get comprehensive database statistics"""
    try:
        if not DB_PATH.exists():
            return {
                "total_articles": 0,
                "total_messages": 0,
                "status": "Database not initialized",
                "database_exists": False
            }
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM articles")
        article_count = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT title, created_at 
            FROM articles 
            ORDER BY created_at DESC 
            LIMIT 5
        """)
        recent_articles = cursor.fetchall()
        
        try:
            cursor.execute("SELECT COUNT(*) FROM chat_history")
            message_count = cursor.fetchone()[0]
        except sqlite3.OperationalError:
            message_count = 0
        
        cursor.execute("SELECT MIN(created_at), MAX(created_at) FROM articles")
        date_range = cursor.fetchone()
        
        conn.close()
        
        stats = {
            "total_articles": article_count,
            "total_messages": message_count,
            "database_exists": True,
            "status": "Active",
            "recent_articles": [
                {"title": title, "date": date} 
                for title, date in recent_articles
            ] if recent_articles else [],
            "date_range": {
                "earliest": date_range[0] if date_range[0] else None,
                "latest": date_range[1] if date_range[1] else None
            }
        }
        
        logger.info(f"Database stats: {article_count} articles, {message_count} messages")
        return stats
        
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return {
            "total_articles": 0,
            "total_messages": 0,
            "status": f"Error: {str(e)}",
            "database_exists": False,
            "error": str(e)
        }


def web_search(query: str, num_results: int = 5) -> Dict[str, Any]:
    """
    Perform web search using Serper API
    
    Returns:
        Dict with 'success', 'results', 'formatted_text', and 'result_count'
    """
    global SERPER_API_KEY
    
    if SERPER_API_KEY is None:
        SERPER_API_KEY = os.getenv("SERPER_API_KEY")
    
    if not SERPER_API_KEY:
        logger.warning("Web search called but SERPER_API_KEY not configured")
        return {
            "success": False,
            "results": [],
            "formatted_text": """⚠️ **Web search is not configured.**

To enable live web search:
1. Get a free API key from https://serper.dev
2. Add `SERPER_API_KEY=your_key_here` to your .env file
3. Restart the application

📌 For now, I can only search the local database.""",
            "result_count": 0,
            "error": "API key not configured"
        }
    
    try:
        url = "https://google.serper.dev/search"
        
        payload = {
            "q": query,
            "num": num_results
        }
        
        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        results = []
        formatted_parts = []
        
        # Add knowledge graph if available
        if "knowledgeGraph" in data:
            kg = data["knowledgeGraph"]
            if "description" in kg:
                formatted_parts.append(f"**📌 Quick Answer:** {kg['description']}")
        
        # Add organic results
        if "organic" in data:
            for idx, result in enumerate(data["organic"][:num_results], 1):
                title = result.get("title", "No title")
                snippet = result.get("snippet", "No description")
                link = result.get("link", "")
                date = result.get("date", "")
                
                results.append({
                    "title": title,
                    "snippet": snippet,
                    "url": link,
                    "date": date,
                    "position": idx
                })
                
                date_str = f" • {date}" if date else ""
                formatted_parts.append(
                    f"{idx}. **{title}**{date_str}\n   {snippet}\n   🔗 {link}"
                )
        
        if results:
            formatted_text = f"🌐 **Web Search Results for '{query}':**\n\n" + "\n\n".join(formatted_parts)
            logger.info(f"Web search successful: {len(results)} results for '{query}'")
            
            return {
                "success": True,
                "results": results,
                "formatted_text": formatted_text,
                "result_count": len(results),
                "query": query
            }
        else:
            return {
                "success": False,
                "results": [],
                "formatted_text": f"No web results found for '{query}'",
                "result_count": 0,
                "query": query
            }
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Web search request failed: {e}")
        return {
            "success": False,
            "results": [],
            "formatted_text": f"❌ Web search failed: {str(e)}",
            "result_count": 0,
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return {
            "success": False,
            "results": [],
            "formatted_text": f"❌ Error during web search: {str(e)}",
            "result_count": 0,
            "error": str(e)
        }


def intelligent_search(query: str, local_sources: list = None) -> Dict[str, Any]:
    """
    DAY 37 CORE FEATURE: Intelligent search with automatic fallback
    
    Implements the logic: 
    - If local search returns < 2 results → automatically trigger web search
    - Combine results intelligently
    
    Args:
        query: Search query
        local_sources: List of local article sources (from get_context)
        
    Returns:
        Dict with combined results and metadata about which sources were used
    """
    result = {
        "query": query,
        "local_count": len(local_sources) if local_sources else 0,
        "web_count": 0,
        "sources_used": [],
        "triggered_web_search": False,
        "formatted_response": "",
        "all_sources": []
    }
    
    # Check if we need web search fallback
    needs_web_search = (
        local_sources is None or 
        len(local_sources) < MIN_LOCAL_RESULTS
    )
    
    if needs_web_search:
        logger.info(
            f"Local results insufficient ({result['local_count']} < {MIN_LOCAL_RESULTS}). "
            f"Triggering web search fallback..."
        )
        result["triggered_web_search"] = True
        result["sources_used"].append("web")
        
        # Execute web search
        web_result = web_search(query, num_results=5)
        result["web_count"] = web_result.get("result_count", 0)
        
        # Build combined response
        response_parts = []
        
        if result["local_count"] > 0:
            response_parts.append(
                f"📚 **Local Database Results ({result['local_count']}):**\n"
            )
            for i, source in enumerate(local_sources, 1):
                response_parts.append(
                    f"{i}. {source.get('title', 'Untitled')}\n"
                    f"   {source.get('description', 'No description')}\n"
                    f"   🔗 {source.get('url', '')}"
                )
            response_parts.append("")  # Blank line separator
        
        # Add web results
        response_parts.append(web_result.get("formatted_text", ""))
        
        result["formatted_response"] = "\n".join(response_parts)
        result["all_sources"] = (local_sources or []) + web_result.get("results", [])
        
    else:
        # Sufficient local results
        logger.info(f"Using {result['local_count']} local results (sufficient)")
        result["sources_used"].append("local")
        
        response_parts = [f"📚 **Local Database Results ({result['local_count']}):**\n"]
        for i, source in enumerate(local_sources, 1):
            response_parts.append(
                f"{i}. **{source.get('title', 'Untitled')}**\n"
                f"   {source.get('description', 'No description')}\n"
                f"   🔗 {source.get('url', '')}"
            )
        
        result["formatted_response"] = "\n".join(response_parts)
        result["all_sources"] = local_sources
    
    return result


def add_article(title: str, description: str, url: str) -> bool:
    """Add a new article to the database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO articles (title, description, url)
            VALUES (?, ?, ?)
        """, (title, description, url))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Added article: {title}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to add article: {e}")
        return False


def get_all_articles() -> list:
    """Retrieve all articles from database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, title, description, url, created_at
            FROM articles
            ORDER BY created_at DESC
        """)
        
        articles = []
        for row in cursor.fetchall():
            articles.append({
                "id": row[0],
                "title": row[1],
                "description": row[2],
                "url": row[3],
                "created_at": row[4]
            })
        
        conn.close()
        return articles
        
    except Exception as e:
        logger.error(f"Failed to retrieve articles: {e}")
        return []


# Testing
if __name__ == "__main__":
    print("=" * 60)
    print("DAY 37: Testing Enhanced Tools with Fallback Logic")
    print("=" * 60)
    
    print("\n1️⃣ Testing get_db_stats():")
    stats = get_db_stats()
    print(f"   Articles: {stats.get('total_articles', 0)}")
    print(f"   Status: {stats.get('status', 'Unknown')}")
    
    print("\n2️⃣ Testing web_search():")
    result = web_search("latest AI breakthroughs 2026", num_results=3)
    print(f"   Success: {result['success']}")
    print(f"   Results: {result['result_count']}")
    if result['success']:
        print(f"   Preview: {result['formatted_text'][:200]}...")
    else:
        print(f"   Message: {result['formatted_text']}")
    
    print("\n3️⃣ Testing intelligent_search() - Fallback Scenario:")
    # Simulate insufficient local results
    mock_local = [{"title": "Single Article", "description": "Not enough", "url": "test.com"}]
    smart_result = intelligent_search("quantum computing news", local_sources=mock_local)
    print(f"   Local count: {smart_result['local_count']}")
    print(f"   Web search triggered: {smart_result['triggered_web_search']}")
    print(f"   Total sources: {len(smart_result['all_sources'])}")
    print(f"   Sources used: {', '.join(smart_result['sources_used'])}")
    
    print("\n✅ Enhanced tools test complete!")
    print("=" * 60)