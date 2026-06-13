import pandas as pd
import sqlite3
import os

def load_to_sqlite():
    """
    Reads the cleaned CSV and loads it into a SQLite database.
    Creates the database file if it doesn't exist.
    """

    # ── Check clean file exists ───────────────────────────────────────────────
    clean_path = "Data/cleaned/startups_clean.csv"
    if not os.path.exists(clean_path):
        print("Clean data not found. Run cleaner.py first.")
        return

    # ── Load CSV ──────────────────────────────────────────────────────────────
    df = pd.read_csv(clean_path)
    print(f"Loaded {len(df)} rows from cleaned CSV")

    # ── Filter out rows where startup_name looks like a URL ───────────────────
    df = df[~df["startup_name"].str.startswith("http")]
    print(f"After URL filter: {len(df)} rows")

    # ── Connect to SQLite ─────────────────────────────────────────────────────
    # This creates the file if it doesn't exist
    os.makedirs("Data", exist_ok=True)
    conn = sqlite3.connect("Data/startups.db")
    print("Connected to Data/startups.db")

    # ── Write to database ─────────────────────────────────────────────────────
    # if_exists="replace" → drops and recreates the table every time
    # This is fine for now; in production you'd use "append" with dedup logic
    df.to_sql("startups", conn, if_exists="replace", index=False)
    print("Written to table: startups")

    # ── Verify by querying back ───────────────────────────────────────────────
    cursor = conn.cursor()

    # Total rows
    cursor.execute("SELECT COUNT(*) FROM startups")
    total = cursor.fetchone()[0]
    print(f"\nTotal rows in DB: {total}")

    # Top 5 sectors by deal count
    print("\nTop 5 sectors by deal count:")
    cursor.execute("""
        SELECT sector, COUNT(*) as deals
        FROM startups
        GROUP BY sector
        ORDER BY deals DESC
        LIMIT 5
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]:<30} {row[1]} deals")

    # Top 5 cities by deal count
    print("\nTop 5 cities by deal count:")
    cursor.execute("""
        SELECT city, COUNT(*) as deals
        FROM startups
        GROUP BY city
        ORDER BY deals DESC
        LIMIT 5
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]:<20} {row[1]} deals")

    conn.close()
    print("\nDatabase connection closed. Pipeline complete.")


if __name__ == "__main__":
    load_to_sqlite()