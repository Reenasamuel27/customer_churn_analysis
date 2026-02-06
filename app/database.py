import sqlite3

conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()

def init_db():
    c.execute("""
    CREATE TABLE IF NOT EXISTS users(
        username TEXT PRIMARY KEY,
        password TEXT,
        role TEXT,
        email TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS logs(
        id INTEGER PRIMARY KEY,
        user TEXT,
        risk REAL,
        time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()

init_db()
