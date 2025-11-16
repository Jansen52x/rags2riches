"""
Generate a Markdown analysis report for the materials agent evaluation.
Mirrors the fact checker analysis workflow.
"""

import json
import os
import sqlite3
from typing import Dict, List

import pandas as pd


def load_results(db_path: str) -> pd.DataFrame:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Evaluation database not found at {db_path}")
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM evaluation_results", conn)
    if df.empty:
        raise ValueError("Evaluation database is empty. Run the evaluator first.")
    return df


def _parse_json_list(series: pd.Series) -> List[List[str]]:
    return series.apply(lambda x: json.loads(x) if isinstance(x, str) and x else []).tolist()


def generate_report(df: pd.DataFrame) -> str:
    df = df.copy()
    df["expected_material_types_list"] = _parse_json_list(df["expected_material_types"])
    df["recommended_material_types_list"] = _parse_json_list(df["recommended_material_types"])
    df["expected_high_priority_types_list"] = _parse_json_list(df["expected_high_priority_types"])
    df["high_priority_matches_list"] = _parse_json_list(df["high_priority_matches"])
    df["extra_material_types_list"] = _parse_json_list(df["extra_material_types"])

    avg_type_coverage = df["type_coverage"].mean()
    avg_high_priority_coverage = df["high_priority_coverage"].mean()
    avg_llm_score = df["llm_judge_score"].mean()
    avg_generated_files = df["generated_file_count"].mean()

    full_coverage = df[df["type_coverage"] >= 0.999]
    partial_coverage = df[(df["type_coverage"] > 0) & (df["type_coverage"] < 0.999)]
    zero_coverage = df[df["type_coverage"] == 0]

    report_lines: List[str] = []
    report_lines.append("# Materials Agent Evaluation Report")
    report_lines.append(f"**Scenarios Evaluated:** {len(df)}")
    report_lines.append("---")
    report_lines.append(f"### 1. Average Type Coverage: **{avg_type_coverage * 100:.2f}%**")
    report_lines.append(f"### 2. High-Priority Coverage: **{avg_high_priority_coverage * 100:.2f}%**")
    report_lines.append(f"### 3. Average LLM Judge Score: **{avg_llm_score:.2f}/100**")
    report_lines.append(f"### 4. Avg. Generated Assets per Scenario: **{avg_generated_files:.2f}**")
    report_lines.append("---")

    report_lines.append("\n## Coverage Breakdown")
    report_lines.append(f"- ✅ Full coverage scenarios: {len(full_coverage)}")
    report_lines.append(f"- ⚠️ Partial coverage scenarios: {len(partial_coverage)}")
    report_lines.append(f"- ❌ Missed coverage scenarios: {len(zero_coverage)}")

    if not zero_coverage.empty:
        report_lines.append("\n### Scenarios Missing Expected Types")
        for _, row in zero_coverage.iterrows():
            report_lines.append(
                f"- `{row['scenario_id']}`: expected {row['expected_material_types_list']} but recommended {row['recommended_material_types_list']}"
            )

    if not partial_coverage.empty:
        report_lines.append("\n### Scenarios With Partial Coverage")
        for _, row in partial_coverage.iterrows():
            report_lines.append(
                f"- `{row['scenario_id']}` ({row['type_coverage']*100:.0f}%): missing {sorted(set(row['expected_material_types_list']) - set(row['recommended_material_types_list']))}"
            )

    report_lines.append("\n## High-Priority Alignment")
    for _, row in df.iterrows():
        expected_high = row["expected_high_priority_types_list"]
        if not expected_high:
            continue
        matched = row["high_priority_matches_list"]
        missing = sorted(set(expected_high) - set(matched))
        line = f"- `{row['scenario_id']}`: matched {matched}"
        if missing:
            line += f" | missing {missing}"
        report_lines.append(line)

    top_llm = df.sort_values(by="llm_judge_score", ascending=False).head(3)
    bottom_llm = df.sort_values(by="llm_judge_score", ascending=True).head(3)

    report_lines.append("\n## Top 3 LLM-Judged Scenarios")
    for _, row in top_llm.iterrows():
        report_lines.append(
            f"- **Score {row['llm_judge_score']:.1f}** | `{row['scenario_id']}` | Types: {row['recommended_material_types_list']}"
        )

    report_lines.append("\n## Lowest 3 LLM-Judged Scenarios")
    for _, row in bottom_llm.iterrows():
        report_lines.append(
            f"- **Score {row['llm_judge_score']:.1f}** | `{row['scenario_id']}` | Reason: {row['llm_judge_reasoning']}"
        )

    report_lines.append("\n## Common Extra Materials Added")
    extra_counts: Dict[str, int] = {}
    for extras in df["extra_material_types_list"]:
        for item in extras:
            extra_counts[item] = extra_counts.get(item, 0) + 1
    if extra_counts:
        for material, count in sorted(extra_counts.items(), key=lambda x: x[1], reverse=True):
            report_lines.append(f"- {material}: {count} scenario(s)")
    else:
        report_lines.append("- None")

    return "\n".join(report_lines)


def save_report(report: str, output_path: str) -> None:
    with open(output_path, "w") as f:
        f.write(report)


def main() -> None:
    here = os.path.dirname(__file__)
    db_path = os.path.join(here, "evaluation_results.db")
    output_path = os.path.join(here, "analysis_report.md")

    df = load_results(db_path)
    report = generate_report(df)
    save_report(report, output_path)
    print(f"Analysis report saved to {output_path}")


if __name__ == "__main__":
    main()
