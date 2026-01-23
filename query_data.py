import sqlite3

def run_queries():
    # Connect to our existing database
    connection = sqlite3.connect('news_data.db')
    cursor = connection.cursor()

    print("--- Query 1: Getting Total Article Count ---")
    cursor.execute('SELECT COUNT(*) FROM articles')
    total = cursor.fetchone()[0]
    print(f"Total articles in DB: {total}\n")

    print("--- Query 2: Listing the first 5 articles ---")
    cursor.execute('SELECT title, source FROM articles LIMIT 5')
    rows = cursor.fetchall()
    for row in rows:
        print(f"Source: {row[1]} | Title: {row[0]}")

    print("\n--- Query 3: Searching for 'Google' in titles ---")
    # Using 'LIKE' and '%' for pattern matching (Search)
    search_term = '%Google%'
    cursor.execute('SELECT title FROM articles WHERE title LIKE ?', (search_term,))
    results = cursor.fetchall()
    
    if results:
        for r in results:
            print(f"Found: {r[0]}")
    else:
        print("No articles found mentioning 'Google'.")

    connection.close()

if __name__ == "__main__":
    run_queries()