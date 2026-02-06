import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import joblib
import plotly.express as px
import plotly.graph_objects as go
import hashlib
import time
import os
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import openai

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Enterprise Executive Dashboard",
    layout="wide"
)

# ================= STYLES =================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top,#020617,#020617,#0f172a,#020617);
    color:white;
}
html, body, [class*="css"] {
    color:#e5e7eb !important;
}
section[data-testid="stSidebar"] * {
    color:white !important;
}
.metric-card {
    background: linear-gradient(145deg,rgba(255,255,255,0.12),rgba(255,255,255,0.02));
    padding:25px;
    border-radius:20px;
    text-align:center;
    backdrop-filter: blur(18px);
    box-shadow: 0 10px 35px rgba(0,255,255,0.15);
}
.stButton>button {
    background: linear-gradient(90deg,#22d3ee,#8b5cf6,#ec4899);
    color:white;
    border-radius:12px;
    padding:10px 28px;
    font-size:16px;
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
model = joblib.load("../models/churn_model.pkl")
features = joblib.load("../models/feature_columns.pkl")

# ================= DATABASE =================
conn = sqlite3.connect("enterprise.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS users(
    username TEXT PRIMARY KEY,
    password TEXT,
    role TEXT
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS logs(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user TEXT,
    risk REAL
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS password_reset_requests(
    username TEXT PRIMARY KEY,
    requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()

# ================= HELPERS =================
def hash_pw(p):
    return hashlib.sha256(p.encode()).hexdigest()

def add_user(u, p, role="user"):
    c.execute(
        "INSERT OR IGNORE INTO users VALUES (?,?,?)",
        (u, hash_pw(p), role)
    )
    conn.commit()

def login(u, p):
    c.execute(
        "SELECT role FROM users WHERE username=? AND password=?",
        (u, hash_pw(p))
    )
    row = c.fetchone()
    return row[0] if row else None

def log_prediction(user, risk):
    c.execute(
        "INSERT INTO logs (user, risk) VALUES (?,?)",
        (user, risk)
    )
    conn.commit()

def get_logs(user=None):
    if user:
        return pd.read_sql_query(
            "SELECT risk FROM logs WHERE user=?",
            conn, params=(user,)
        )
    return pd.read_sql_query("SELECT risk FROM logs", conn)

def request_password_reset(username):
    c.execute(
        "INSERT OR IGNORE INTO password_reset_requests(username) VALUES (?)",
        (username,)
    )
    conn.commit()

def get_reset_requests():
    return pd.read_sql_query(
        "SELECT * FROM password_reset_requests",
        conn
    )

def reset_user_password(username, new_password):
    c.execute(
        "UPDATE users SET password=? WHERE username=?",
        (hash_pw(new_password), username)
    )
    c.execute(
        "DELETE FROM password_reset_requests WHERE username=?",
        (username,)
    )
    conn.commit()

def update_user_role(username, role):
    c.execute(
        "UPDATE users SET role=? WHERE username=?",
        (role, username)
    )
    conn.commit()

# ================= DEFAULT ADMIN =================
add_user("admin", "admin123", "admin")

# ================= SESSION =================
if "user" not in st.session_state:
    st.session_state.user = None
if "role" not in st.session_state:
    st.session_state.role = None

# ================= LOGIN =================
if not st.session_state.user:

    st.title("🔐 Enterprise Login Portal")

    tab1, tab2, tab3 = st.tabs(["Login", "Register", "Forgot Password"])

    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            role = login(u, p)
            if role:
                st.session_state.user = u
                st.session_state.role = role
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        nu = st.text_input("New Username")
        npw = st.text_input("New Password", type="password")
        if st.button("Create Account"):
            add_user(nu, npw)
            st.success("Account created")

    with tab3:
        fu = st.text_input("Username for reset")
        if st.button("Request Password Reset"):
            c.execute("SELECT username FROM users WHERE username=?", (fu,))
            if c.fetchone():
                request_password_reset(fu)
                st.success("Reset request sent to admin")
            else:
                st.error("Username not found")

    st.stop()

# ================= SIDEBAR =================
st.sidebar.success(f"User: {st.session_state.user}")
st.sidebar.info(f"Role: {st.session_state.role}")

if st.sidebar.button("Logout"):
    st.session_state.user = None
    st.session_state.role = None
    st.rerun()

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Predict", "Segmentation", "Anomaly Detection", "Admin"]
)

logs = get_logs(st.session_state.user)

# ================= DASHBOARD =================
if page == "Dashboard":

    st.title("📊 Enterprise Executive Dashboard")

    avg = logs["risk"].mean() if not logs.empty else 0
    total = len(logs)
    high = (logs["risk"] > 0.5).sum() if not logs.empty else 0

    cols = st.columns(3)
    cards = [
        ("Predictions", total),
        ("Average Risk", f"{avg:.2f}"),
        ("High Risk", high)
    ]

    for col, (l, v) in zip(cols, cards):
        col.markdown(
            f'<div class="metric-card"><h2>{v}</h2><p>{l}</p></div>',
            unsafe_allow_html=True
        )

    if not logs.empty:
        st.subheader("📈 Risk Analytics")

        c1, c2 = st.columns(2)

        with c1:
            fig1 = px.histogram(logs, x="risk", title="Risk Distribution")
            st.plotly_chart(fig1, use_container_width=True)

        with c2:
            fig2 = px.line(logs, y="risk", title="Risk Trend")
            st.plotly_chart(fig2, use_container_width=True)

        c3, c4 = st.columns(2)

        with c3:
            split = pd.DataFrame({
                "Type": ["High Risk", "Low Risk"],
                "Count": [
                    (logs["risk"] > 0.5).sum(),
                    (logs["risk"] <= 0.5).sum()
                ]
            })
            fig3 = px.pie(split, names="Type", values="Count", hole=0.4)
            st.plotly_chart(fig3, use_container_width=True)

        with c4:
            fig4 = px.box(logs, y="risk", title="Risk Variability")
            st.plotly_chart(fig4, use_container_width=True)

# ================= PREDICT =================
if page == "Predict":

    st.title("🤖 Churn Prediction Engine")

    c1, c2, c3 = st.columns(3)
    tenure = c1.slider("Tenure", 0, 72, 12)
    monthly = c2.slider("Monthly Charges", 20, 120, 70)
    total_c = c3.slider("Total Charges", 20, 100000, 1000)

    input_df = pd.DataFrame(
        np.zeros((1, len(features))),
        columns=features
    )
    input_df["tenure"] = tenure
    input_df["MonthlyCharges"] = monthly
    input_df["TotalCharges"] = total_c

    if st.button("Predict Risk"):
        risk = model.predict_proba(input_df)[0][1]
        log_prediction(st.session_state.user, risk)

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=risk * 100,
                title={"text": "Churn Risk %"}
            )
        )
        st.plotly_chart(fig, use_container_width=True)

# ================= SEGMENTATION =================
if page == "Segmentation":

    st.title("👥 Customer Segmentation")

    if logs.empty:
        st.info("Run predictions first")
    else:
        km = KMeans(n_clusters=3, random_state=42)
        logs["segment"] = km.fit_predict(logs[["risk"]])

        fig = px.scatter(
            logs,
            x=logs.index,
            y="risk",
            color="segment",
            title="Risk Segments"
        )
        st.plotly_chart(fig, use_container_width=True)

# ================= ANOMALY =================
if page == "Anomaly Detection":

    st.title("🚨 Anomaly Detection")

    if logs.empty:
        st.info("No data yet")
    else:
        iso = IsolationForest(contamination=0.1, random_state=42)
        logs["anomaly"] = iso.fit_predict(logs[["risk"]])

        fig = px.scatter(
            logs,
            x=logs.index,
            y="risk",
            color="anomaly",
            title="Anomaly Detection"
        )
        st.plotly_chart(fig, use_container_width=True)


# ================= ADMIN =================
if page == "Admin":

    if st.session_state.role != "admin":
        st.error("Admin only")
        st.stop()

    st.title("🛠 Admin Control Panel")

    tab1, tab2 = st.tabs(["👥 Users", "📊 System Analytics"])

    with tab1:
        users = pd.read_sql_query("SELECT username, role FROM users", conn)
        st.dataframe(users, use_container_width=True)

        st.subheader("Add User")
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        r = st.selectbox("Role", ["user", "admin"])
        if st.button("Create User"):
            add_user(u, p, r)
            st.success("User added")

        st.divider()
        st.subheader("🔐 Password Reset Requests")
        reqs = get_reset_requests()

        if reqs.empty:
            st.info("No reset requests")
        else:
            for _, row in reqs.iterrows():
                new_pw = st.text_input(
                    f"New password for {row['username']}",
                    type="password",
                    key=row['username']
                )
                if st.button(f"Reset {row['username']}"):
                    reset_user_password(row['username'], new_pw)
                    st.success("Password reset done")
                    st.rerun()

        st.divider()
        st.subheader("🛡 Role Management")
        su = st.selectbox("Select User", users["username"])
        nr = st.selectbox("Assign Role", ["user", "admin"])
        if st.button("Update Role"):
            update_user_role(su, nr)
            st.success("Role updated")
            st.rerun()

    with tab2:
        all_logs = get_logs()
        st.metric("Total Predictions", len(all_logs))
        st.metric("Average Risk", round(all_logs["risk"].mean(), 2))

        fig = px.histogram(all_logs, x="risk", title="System Risk Distribution")
        st.plotly_chart(fig, use_container_width=True)