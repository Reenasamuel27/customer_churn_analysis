import hashlib
from database import conn, c

def hash_pw(p):
    return hashlib.sha256(p.encode()).hexdigest()

def register(u,p,email,role="user"):
    c.execute("INSERT OR IGNORE INTO users VALUES (?,?,?,?)",
              (u,hash_pw(p),role,email))
    conn.commit()

def login(u,p):
    c.execute("SELECT role FROM users WHERE username=? AND password=?",
              (u,hash_pw(p)))
    return c.fetchone()

def reset_password(email,newp):
    c.execute("UPDATE users SET password=? WHERE email=?",
              (hash_pw(newp),email))
    conn.commit()
