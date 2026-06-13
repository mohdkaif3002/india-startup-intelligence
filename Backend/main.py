from fastapi import FastAPI, Query
import sqlite3
import pandas as pd
import joblib
import numpy as np

app = FastAPI(
    title="India Startup Funding Intelligence",
    description="API for exploring India startup funding data from 2015-2024",
    version="1.0.0"
)

DB_PATH = "Data/startups.db"


def query_db(sql: str, params: tuple = ()):
    """Helper function — runs any SQL query and returns results as a list of dicts."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # makes rows behave like dicts
    cursor = conn.cursor()
    cursor.execute(sql, params)
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return results


# ── ENDPOINT 1 — Health check ─────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "India Startup Intelligence API is running"}


# ── ENDPOINT 2 — Get startups with filters ────────────────────────────────────
@app.get("/startups")
def get_startups(
    sector: str = Query(None, description="Filter by sector e.g. FinTech"),
    city:   str = Query(None, description="Filter by city e.g. Bangalore"),
    limit:  int = Query(50,   description="Number of results to return")
):
    """
    Returns a list of startups.
    Optional filters: sector, city
    """
    sql = "SELECT * FROM startups WHERE 1=1"
    params = []

    if sector:
        sql += " AND sector LIKE ?"
        params.append(f"%{sector}%")

    if city:
        sql += " AND city LIKE ?"
        params.append(f"%{city}%")

    sql += " LIMIT ?"
    params.append(limit)

    results = query_db(sql, tuple(params))
    return {"count": len(results), "data": results}


# ── ENDPOINT 3 — Funding trends by year ──────────────────────────────────────
@app.get("/trends")
def get_trends():
    """Returns total funding and deal count grouped by year."""
    sql = """
        SELECT
            SUBSTR(date, -4) as year,
            COUNT(*) as deals,
            ROUND(SUM(amount_usd) / 1000000, 2) as total_funding_million_usd
        FROM startups
        WHERE amount_usd IS NOT NULL
        GROUP BY year
        ORDER BY year
    """
    results = query_db(sql)
    return {"data": results}


# ── ENDPOINT 4 — Top sectors ──────────────────────────────────────────────────
@app.get("/sectors")
def get_sectors():
    """Returns top sectors by number of deals."""
    sql = """
        SELECT
            sector,
            COUNT(*) as deals,
            ROUND(AVG(amount_usd) / 1000000, 2) as avg_funding_million_usd
        FROM startups
        WHERE sector != 'nan'
        GROUP BY sector
        ORDER BY deals DESC
        LIMIT 15
    """
    results = query_db(sql)
    return {"data": results}


# ── ENDPOINT 5 — Top cities ───────────────────────────────────────────────────
@app.get("/cities")
def get_cities():
    """Returns top cities by number of deals."""
    sql = """
        SELECT
            city,
            COUNT(*) as deals,
            ROUND(SUM(amount_usd) / 1000000, 2) as total_funding_million_usd
        FROM startups
        WHERE city != 'nan'
        GROUP BY city
        ORDER BY deals DESC
        LIMIT 10
    """
    results = query_db(sql)
    return {"data": results}

# ── ENDPOINT 6 — Predict fundraising likelihood ───────────────────────────────
model    = joblib.load("Model/model.pkl")
encoders = joblib.load("Model/encoders.pkl")


@app.get("/predict")
def predict(
    sector:          str = Query(..., description="e.g. FinTech"),
    city:            str = Query(..., description="e.g. Bangalore"),
    investment_type: str = Query(..., description="e.g. Series A"),
    amount_usd:      float = Query(0,  description="Amount raised so far in USD")
):
    """
    Predicts whether a startup is likely to raise funding again.
    Returns probability score between 0 and 1.
    """
    try:
        # Encode inputs using saved encoders
        def encode(col, val):
            le = encoders[col]
            # normalize casing — try exact, then title case, then lower, then upper
            for v in [val, val.title(), val.lower(), val.upper()]:
                if v in le.classes_:
                    return le.transform([v])[0]
            return 0  # unknown category defaults to 0

        features = np.array([[
            encode("sector", sector),
            encode("city", city),
            encode("investment_type", investment_type),
            amount_usd
        ]])

        prob  = model.predict_proba(features)[0][1]
        label = "Likely to raise again" if prob >= 0.5 else "Unlikely to raise again"

        return {
            "sector":          sector,
            "city":            city,
            "investment_type": investment_type,
            "amount_usd":      amount_usd,
            "probability":     round(float(prob), 3),
            "prediction":      label
        }

    except Exception as e:
        return {"error": str(e)}