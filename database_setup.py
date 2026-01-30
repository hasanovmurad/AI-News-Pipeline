import sqlite3

# 1. Connect to the database (Creates the file if it doesn't exist)
# Using 'news_data.db' as our local database file
connection = sqlite3.connect('news_data.db')
cursor = connection.cursor()

# 2. Create the 'articles' table using SQL
# We define specific columns and their data types (TEXT, INTEGER)
print("Creating table if it does not exist...")
cursor.execute('''
    CREATE TABLE IF NOT EXISTS articles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        source TEXT,
        date TEXT,
        url TEXT
    )
''')

# 3. Test: Inserting a sample record
# Using '?' placeholders is a security best practice (prevents SQL Injection)
sample_data = ("AI Trends at BTU", "Cottbus Tech Journal", "2024-05-20", "https://b-tu.de")

cursor.execute('''
    INSERT INTO articles (title, source, date, url)
    VALUES (?, ?, ?, ?)
''', sample_data)

# 4. Commit changes and close the connection
connection.commit()
connection.close()

conn = sqlite3.connect('news_data.db')
conn.execute('''CREATE TABLE IF NOT EXISTS chat_history 
                (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                 role TEXT, 
                 content TEXT, 
                 timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
conn.close()

print("Database initialized and sample data inserted successfully!")