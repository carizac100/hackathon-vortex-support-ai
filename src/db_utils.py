import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path("support_ai.db")

def get_connection():
    return sqlite3.connect(DB_PATH)

def insert_raw_ticket(client_name, project_name, channel, original_text):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO raw_tickets (created_at, client_name, project_name, channel, original_text)
        VALUES (?, ?, ?, ?, ?)
    """, (datetime.utcnow().isoformat(), client_name, project_name, channel, original_text))
    conn.commit()
    ticket_id = cur.lastrowid
    conn.close()
    return ticket_id

def insert_core_ticket(ticket_id, cleaned_text, is_phishing, has_pii, sentiment_score, word_count):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO core_tickets_enriched
        (ticket_id, cleaned_text, is_phishing, has_pii, sentiment_score, word_count)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (ticket_id, cleaned_text, is_phishing, has_pii, sentiment_score, word_count))
    conn.commit()
    conn.close()

def insert_gold_prediction(ticket_id, ticket_type_pred, churn_risk_pred, risk_segment, recommendation_text):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO gold_ticket_predictions
        (ticket_id, ticket_type_pred, churn_risk_pred, risk_segment, recommendation_text)
        VALUES (?, ?, ?, ?, ?)
    """, (ticket_id, ticket_type_pred, churn_risk_pred, risk_segment, recommendation_text))
    conn.commit()
    conn.close()
