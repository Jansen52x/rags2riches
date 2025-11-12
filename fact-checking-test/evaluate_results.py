import sqlite3
import pandas as pd
import os

def analyze_evaluation_results(db_path: str = 'evaluation_results.db', output_path: str = 'analysis_report.md'):
    """
    Analyzes the evaluation results from the SQLite database and saves it to a file.

    Args:
        db_path: Path to the SQLite database file.
        output_path: Path to save the analysis report.
    """
    report_lines = []

    if not os.path.exists(db_path):
        error_msg = f"Error: Database file not found at '{db_path}'"
        print(error_msg)
        report_lines.append(error_msg)
        return

    try:
        # Connect to the SQLite database
        with sqlite3.connect(db_path) as conn:
            # Load the data into a pandas DataFrame
            df = pd.read_sql_query("SELECT * FROM evaluation_results", conn)
    except Exception as e:
        error_msg = f"Error reading the database: {e}"
        print(error_msg)
        report_lines.append(error_msg)
        return

    if df.empty:
        msg = "The database is empty. No results to analyze."
        print(msg)
        report_lines.append(msg)
        return

    # --- Calculate Key Metrics ---
    total_claims = len(df)
    exact_matches = df['exact_match'].sum()
    exact_match_accuracy = (exact_matches / total_claims) * 100 if total_claims > 0 else 0
    avg_llm_score = df['llm_judge_score'].mean()

    # --- Build Report ---
    report_lines.append("# Evaluation Analysis Report")
    report_lines.append(f"**Total Claims Analyzed:** {total_claims}")
    report_lines.append("---")
    report_lines.append(f"### 1. Exact Match Accuracy: **{exact_match_accuracy:.2f}%** ({exact_matches}/{total_claims})")
    report_lines.append(f"### 2. Average LLM Judge Score: **{avg_llm_score:.2f}/100**")
    report_lines.append("---")

    # --- General Insights ---
    report_lines.append("\n## Insights & Observations")

    # 1. Best and Worst Performing Claims
    report_lines.append("\n### Top 3 Highest-Scoring Claims (by LLM Judge)")
    top_claims = df.sort_values(by='llm_judge_score', ascending=False).head(3)
    for _, row in top_claims.iterrows():
        report_lines.append(f"- **Score: {row['llm_judge_score']:.1f}** | Claim ID: `{row['claim_id']}` | Claim: *'{row['claim'][:50]}...'*")

    report_lines.append("\n### Top 3 Lowest-Scoring Claims (by LLM Judge)")
    bottom_claims = df.sort_values(by='llm_judge_score', ascending=True).head(3)
    for _, row in bottom_claims.iterrows():
        report_lines.append(f"- **Score: {row['llm_judge_score']:.1f}** | Claim ID: `{row['claim_id']}` | Claim: *'{row['claim'][:50]}...'*")
        report_lines.append(f"  - **LLM Reason:** {row['llm_judge_reasoning']}")


    # 2. Mismatched Verdicts Analysis
    mismatched_verdicts = df[df['exact_match'] == False]
    if not mismatched_verdicts.empty:
        report_lines.append(f"\n## Analysis of {len(mismatched_verdicts)} Incorrect Verdicts")
        for i, row in mismatched_verdicts.iterrows():
            report_lines.append(f"\n**{i+1}. Claim ID: `{row['claim_id']}`**")
            report_lines.append(f"   - **Claim:** *'{row['claim']}'*")
            report_lines.append(f"   - **Expected:** `{row['expected_verdict']}` | **Agent returned:** `{row['actual_verdict']}`")
            report_lines.append(f"   - **LLM Score:** {row['llm_judge_score']:.1f}")
            report_lines.append(f"   - **LLM Reason:** {row['llm_judge_reasoning']}")
    else:
        report_lines.append("\n## No incorrect verdicts found. Perfect exact match accuracy!")

    # --- Save and Print Report ---
    report_content = "\n".join(report_lines)
    try:
        with open(output_path, 'w') as f:
            f.write(report_content)
        print(f"\nAnalysis report saved to '{output_path}'")
    except Exception as e:
        print(f"Error saving report file: {e}")

    print("\n" + report_content)


if __name__ == "__main__":
    # The script assumes the DB is in the same directory.
    # If it's located elsewhere, provide the full path.
    analyze_evaluation_results('evaluation_results.db', 'analysis_report.md')
