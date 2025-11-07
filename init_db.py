import os
import psycopg

# Allow configuration via environment variables (used in docker-compose db_init)
DB_NAME = os.getenv("POSTGRES_DB", "claim_verifications")
DB_USER = os.getenv("POSTGRES_USER", "fact-checker")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "fact-checker")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = int(os.getenv("POSTGRES_PORT", "5432"))

conn_str = f"dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD} host={DB_HOST} port={DB_PORT}"

with psycopg.connect(conn_str) as conn:
    conn.execute(
        """
        CREATE EXTENSION IF NOT EXISTS "pgcrypto";

        CREATE TABLE IF NOT EXISTS claim_verifications (
            original_claim TEXT NOT NULL,
            original_claim_id TEXT NOT NULL,
            verification_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            salesperson_id TEXT NOT NULL,
            overall_verdict BOOLEAN NOT NULL,
            explanation TEXT,
            main_evidence JSONB,
            pass_to_materials_agent BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
    )

    print("Table 'claim_verifications' is ready.")
