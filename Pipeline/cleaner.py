import pandas as pd
import os
import re

# ── 1. LOAD ───────────────────────────────────────────────────────────────────

def load_all_datasets():
    """Load all three raw CSVs into separate DataFrames."""

    df1 = pd.read_csv("Data/raw/startup_funding.csv")
    print(f"startup_funding.csv        → {df1.shape[0]} rows")

    df2 = pd.read_csv("Data/raw/Indian_startup_funding_2023.csv")
    print(f"Indian_startup_funding_2023 → {df2.shape[0]} rows")

    df3 = pd.read_csv("Data/raw/Startup_funding_2024.csv")
    print(f"Startup_funding_2024.csv   → {df3.shape[0]} rows")

    return df1, df2, df3


# ── 2. STANDARDIZE COLUMNS ────────────────────────────────────────────────────

def standardize(df1, df2, df3):
    """
    Rename all columns to a unified standard:
    startup_name, sector, city, amount_usd, investment_type, investors, date
    """

    # df1 and df2 have same structure
    rename_map_12 = {
        "Startup Name":     "startup_name",
        "Industry Vertical":"sector",
        "City  Location":   "city",
        "Amount in USD":    "amount_usd",
        "InvestmentnType":  "investment_type",   # df1 typo
        "Investment Type":  "investment_type",   # df2 correct
        "Investors Name":   "investors",
        "Date dd/mm/yyyy":  "date",
    }

    df1 = df1.rename(columns=rename_map_12)
    df2 = df2.rename(columns=rename_map_12)

    # df3 has different column names
    rename_map_3 = {
        "Company":           "startup_name",
        "Sector":            "sector",
        "Headquarters":      "city",
        "Amount":            "amount_usd",
        "Funding_Round_Type":"investment_type",
        "Lead_Investors":    "investors",
    }
    df3 = df3.rename(columns=rename_map_3)
    df3["date"] = "2024"   # no date column in 2024 dataset

    # Keep only the standard columns
    standard_cols = ["startup_name", "sector", "city",
                     "amount_usd", "investment_type", "investors", "date"]

    df1 = df1[standard_cols]
    df2 = df2[standard_cols]
    df3 = df3[standard_cols]

    print("\nAll three datasets standardized to same columns.")
    return df1, df2, df3


# ── 3. COMBINE ────────────────────────────────────────────────────────────────

def combine(df1, df2, df3):
    """Stack all three DataFrames into one."""
    df = pd.concat([df1, df2, df3], ignore_index=True)
    print(f"\nCombined total rows (before dedup): {len(df)}")
    return df


# ── 4. CLEAN ──────────────────────────────────────────────────────────────────

def clean(df):
    """
    Full cleaning pipeline:
    1. Deduplicate
    2. Strip whitespace
    3. Fix amount column
    4. Standardize city names
    5. Drop rows missing critical fields
    """

    # Step 1 — Deduplicate on startup_name + amount_usd + date
    before = len(df)
    df = df.drop_duplicates(subset=["startup_name", "amount_usd", "date"])
    print(f"Duplicates removed: {before - len(df)}")

    # Step 2 — Strip whitespace from all string columns
    str_cols = ["startup_name", "sector", "city", "investment_type", "investors"]
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    # Step 3 — Clean amount column
    # Remove commas, currency symbols, convert to numeric
    df["amount_usd"] = (
        df["amount_usd"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.strip()
    )
    df["amount_usd"] = pd.to_numeric(df["amount_usd"], errors="coerce")

    # Step 4 — Standardize common city name variations
    city_map = {
        "Bengaluru": "Bangalore",
        "bengaluru": "Bangalore",
        "bangalore": "Bangalore",
        "new delhi": "Delhi",
        "New Delhi": "Delhi",
        "delhi": "Delhi",
        "mumbai": "Mumbai",
        "hyderabad": "Hyderabad",
        "chennai": "Chennai",
        "pune": "Pune",
    }
    df["city"] = df["city"].replace(city_map)

    # Step 5 — Drop rows missing startup name or sector
    before = len(df)
    df = df.dropna(subset=["startup_name", "sector"])
    df = df[df["startup_name"] != "nan"]
    df = df[df["sector"] != "nan"]
    print(f"Rows dropped (missing name/sector): {before - len(df)}")

    print(f"\nFinal clean rows: {len(df)}")
    return df


# ── 5. SAVE ───────────────────────────────────────────────────────────────────

def save(df):
    os.makedirs("Data/cleaned", exist_ok=True)
    path = "Data/cleaned/startups_clean.csv"
    df.to_csv(path, index=False)
    print(f"\nSaved to {path}")
    print("\nPreview:")
    print(df.head())
    print("\nColumn info:")
    print(df.dtypes)
    print(f"\nAmount stats:")
    print(df["amount_usd"].describe())


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df1, df2, df3 = load_all_datasets()
    df1, df2, df3 = standardize(df1, df2, df3)
    df         = combine(df1, df2, df3)
    df         = clean(df)
    save(df)