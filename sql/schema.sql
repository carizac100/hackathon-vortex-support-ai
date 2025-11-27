-- RAW: tickets originales
CREATE TABLE IF NOT EXISTS raw_tickets (
    ticket_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at     TEXT,
    client_name    TEXT,
    project_name   TEXT,
    channel        TEXT,   -- email, portal, whatsapp
    original_text  TEXT
);

-- CORE: texto enriquecido y seguro
CREATE TABLE IF NOT EXISTS core_tickets_enriched (
    ticket_id        INTEGER PRIMARY KEY,
    cleaned_text     TEXT,
    is_phishing      INTEGER,  -- 0/1
    has_pii          INTEGER,  -- 0/1
    sentiment_score  REAL,     -- -1 a 1
    word_count       INTEGER,
    FOREIGN KEY (ticket_id) REFERENCES raw_tickets(ticket_id)
);

-- GOLD: resultados listos para negocio
CREATE TABLE IF NOT EXISTS gold_ticket_predictions (
    ticket_id           INTEGER PRIMARY KEY,
    ticket_type_pred    TEXT,   -- Correctivo / Evolutivo
    churn_risk_pred     REAL,   -- 0 a 100
    risk_segment        TEXT,   -- Bajo / Medio / Alto
    recommendation_text TEXT,
    FOREIGN KEY (ticket_id) REFERENCES raw_tickets(ticket_id)
);
