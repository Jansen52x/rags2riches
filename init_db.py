import psycopg

with psycopg.connect(
    f"dbname=claim_verifications user=fact-checker password=fact-checker host=localhost port=5432"
) as conn:
    conn.execute("""
        CREATE EXTENSION IF NOT EXISTS "pgcrypto";

        CREATE TABLE IF NOT EXISTS claim_verifications (
            verification_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            salesperson_id TEXT NOT NULL,
            claim_id TEXT NOT NULL,
            overall_verdict BOOLEAN NOT NULL,
            explanation TEXT,
            main_evidence JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)
    
    print("Table 'claim_verifications' is ready.")
