import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

def create_source_chart():
    # 1. Connect to our database
    conn = sqlite3.connect('news_data.db')
    
    # 2. Use SQL to get the count of articles per source
    query = """
    SELECT source, COUNT(*) as count 
    FROM articles 
    GROUP BY source 
    ORDER BY count DESC 
    LIMIT 10
    """
    
    # 3. Load the result directly into a Pandas DataFrame
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("No data found to visualize!")
        return

    # 4. Create the Plot
    plt.figure(figsize=(10, 6))
    plt.bar(df['source'], df['count'], color='skyblue')
    
    # Adding professional labels (English only)
    plt.title('Top 10 News Sources for AI', fontsize=14)
    plt.xlabel('News Source', fontsize=12)
    plt.ylabel('Number of Articles', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # 5. Save the chart as an image
    plt.tight_layout()
    plt.savefig('source_distribution.png')
    print("Chart saved successfully as 'source_distribution.png'!")

if __name__ == "__main__":
    create_source_chart()