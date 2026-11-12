import sqlite3
import os
import pandas as pd
from src.config import Config

def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(Config.DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS discovered_influencers (
        rank INTEGER,
        profile_id TEXT PRIMARY KEY,
        reference_profile_name TEXT,
        name TEXT,
        linkedin_url TEXT,
        current_role TEXT,
        organisation TEXT,
        location TEXT,
        persona_category TEXT,
        experience_years INTEGER,
        schools_impacted_estimate INTEGER,
        boards_expertise TEXT,
        ecosystem_indicators TEXT,
        semantic_similarity_score REAL,
        ecosystem_authority_score REAL,
        persona_alignment_score REAL,
        relevance_score REAL,
        match_explanation TEXT
    )
    """)
    conn.commit()
    conn.close()

def save_ranked_results(reference_name: str, ranked_list: list[dict]):
    init_db()
    conn = sqlite3.connect(Config.DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM discovered_influencers")

    for row in ranked_list:
        cursor.execute("""
        INSERT INTO discovered_influencers (
            rank, profile_id, reference_profile_name, name, linkedin_url,
            current_role, organisation, location, persona_category,
            experience_years, schools_impacted_estimate, boards_expertise,
            ecosystem_indicators, semantic_similarity_score,
            ecosystem_authority_score, persona_alignment_score,
            relevance_score, match_explanation
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row["rank"], row["profile_id"], reference_name, row["name"], row["linkedin_url"],
            row["current_role"], row["organisation"], row["location"], row["persona_category"],
            row["experience_years"], row["schools_impacted_estimate"], row["boards_expertise"],
            row["ecosystem_indicators"], row["semantic_similarity_score"],
            row["ecosystem_authority_score"], row["persona_alignment_score"],
            row["relevance_score"], row["match_explanation"]
        ))

    conn.commit()
    df = pd.read_sql_query("SELECT * FROM discovered_influencers ORDER BY rank ASC", conn)
    conn.close()
    df.to_csv("data/ranked_school_influencers.csv", index=False)
