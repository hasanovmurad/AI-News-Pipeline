import sqlite3

def remove_duplicates():
    conn = sqlite3.connect('news_data.db')
    cursor = conn.cursor()
    
    # This SQL command finds duplicates by title and keeps only the unique ones
    cursor.execute('''
        DELETE FROM articles 
        WHERE id NOT IN (
            SELECT MIN(id) 
            FROM articles 
            GROUP BY title
        )
    ''')
    
    print(f"Done! Removed {cursor.rowcount} duplicate rows from the database.")
    conn.commit()
    conn.close()

if __name__ == "__main__":
    remove_duplicates()