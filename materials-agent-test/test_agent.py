"""
Materials Decision Agent Evaluation Script

This script runs the evaluation pipeline for the
materials decision workflow. It runs predefined scenarios, records metrics
in a SQLite database, and leverages Gemini to judge the quality of the
recommendations.

Usage:
    python materials-agent/evaluation/test_agent.py

Environment:
    Requires GOOGLE_API_KEY to be set (e.g. via secrets.env).
"""

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import google.generativeai as genai
from dotenv import load_dotenv
from typing_extensions import TypedDict

# Ensure project root is on the Python path when running from repo root.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in os.sys.path:
    os.sys.path.insert(0, PROJECT_ROOT)

from fast_api.agents.materials_decision_agent import (  # type: ignore
    MaterialsDecisionState,
    create_materials_decision_workflow,
)

# Load environment variables from secrets if present
load_dotenv(os.path.join(PROJECT_ROOT, "secrets.env"))
load_dotenv(os.path.join(PROJECT_ROOT, "fast_api", "secrets.env"))

# Force heuristic mode for deterministic evaluations unless explicitly overridden
os.environ.setdefault("MATERIALS_AGENT_MODE", "heuristic")


class ScenarioConfig(TypedDict):
    scenario_id: str
    description: str
    client_context: str
    verified_claims: List[Dict]
    expected_material_types: List[str]
    expected_high_priority_types: List[str]


@dataclass
class ScenarioEvaluationResult:
    scenario_id: str
    description: str
    expected_material_types: List[str]
    recommended_material_types: List[str]
    expected_high_priority_types: List[str]
    high_priority_matches: List[str]
    type_coverage: float
    high_priority_coverage: float
    extra_material_types: List[str]
    total_recommendations: int
    selected_material_count: int
    generated_file_count: int
    llm_judge_score: float
    llm_judge_reasoning: str


class MaterialsAgentEvaluator:
    def __init__(self, gemini_api_key: Optional[str] = None, db_path: str = "evaluation_results.db") -> None:
        api_key = gemini_api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key required. Set GOOGLE_API_KEY environment variable.")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        self.db_path = db_path
        self._init_db()

    # Database helpers -------------------------------------------------
    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS evaluation_results (
                    scenario_id TEXT PRIMARY KEY,
                    description TEXT,
                    expected_material_types TEXT,
                    recommended_material_types TEXT,
                    expected_high_priority_types TEXT,
                    high_priority_matches TEXT,
                    type_coverage REAL,
                    high_priority_coverage REAL,
                    extra_material_types TEXT,
                    total_recommendations INTEGER,
                    selected_material_count INTEGER,
                    generated_file_count INTEGER,
                    llm_judge_score REAL,
                    llm_judge_reasoning TEXT
                )
                """
            )
            conn.commit()

    def _save_result_to_db(self, result: ScenarioEvaluationResult) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO evaluation_results (
                    scenario_id,
                    description,
                    expected_material_types,
                    recommended_material_types,
                    expected_high_priority_types,
                    high_priority_matches,
                    type_coverage,
                    high_priority_coverage,
                    extra_material_types,
                    total_recommendations,
                    selected_material_count,
                    generated_file_count,
                    llm_judge_score,
                    llm_judge_reasoning
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.scenario_id,
                    result.description,
                    json.dumps(result.expected_material_types),
                    json.dumps(result.recommended_material_types),
                    json.dumps(result.expected_high_priority_types),
                    json.dumps(result.high_priority_matches),
                    result.type_coverage,
                    result.high_priority_coverage,
                    json.dumps(result.extra_material_types),
                    result.total_recommendations,
                    result.selected_material_count,
                    result.generated_file_count,
                    result.llm_judge_score,
                    result.llm_judge_reasoning,
                ),
            )
            conn.commit()

    # Evaluation pipeline ----------------------------------------------
    def load_scenarios(self, dataset_path: str) -> List[ScenarioConfig]:
        with open(dataset_path, "r") as f:
            raw = json.load(f)
        scenarios: List[ScenarioConfig] = []
        for item in raw:
            scenarios.append(
                ScenarioConfig(
                    scenario_id=item["scenario_id"],
                    description=item["description"],
                    client_context=item["client_context"],
                    verified_claims=item["verified_claims"],
                    expected_material_types=item.get("expected_material_types", []),
                    expected_high_priority_types=item.get("expected_high_priority_types", []),
                )
            )
        return scenarios

    def evaluate(self, dataset_path: str) -> Dict[str, float]:
        scenarios = self.load_scenarios(dataset_path)
        workflow = create_materials_decision_workflow()

        results: List[ScenarioEvaluationResult] = []
        total_type_coverage = 0.0
        total_high_priority_coverage = 0.0
        total_llm_score = 0.0

        print(f"\n{'=' * 90}")
        print(f"MATERIALS AGENT EVALUATION - {len(scenarios)} scenarios")
        print(f"{'=' * 90}\n")

        for scenario in scenarios:
            print(f"Evaluating scenario: {scenario['scenario_id']} - {scenario['description']}")

            initial_state: MaterialsDecisionState = MaterialsDecisionState(
                session_id=str(uuid.uuid4()),
                salesperson_id="EVAL",
                client_context=scenario["client_context"],
                user_prompt=None,
                verified_claims=scenario["verified_claims"],
                material_recommendations=[],
                selected_materials=[],
                generation_queue=[],
                generated_files=[],
                generation_status=None,
                decision_complete=False,
                user_feedback=None,
            )

            final_state = workflow.invoke(initial_state)

            recommendations = final_state.get("material_recommendations", [])
            selected = final_state.get("selected_materials", [])
            generated_files = final_state.get("generated_files", [])

            recommended_types = [rec.get("material_type", "") for rec in recommendations]
            expected_types = scenario["expected_material_types"]

            type_hits = sum(1 for t in expected_types if t in recommended_types)
            type_coverage = (type_hits / len(expected_types)) if expected_types else 1.0

            expected_high = scenario["expected_high_priority_types"]
            high_priority_matches = [
                rec.get("material_type")
                for rec in recommendations
                if rec.get("material_type") in expected_high and rec.get("priority", "").lower() == "high"
            ]
            high_priority_coverage = (
                len(high_priority_matches) / len(expected_high) if expected_high else 1.0
            )

            extra_types = sorted({t for t in recommended_types if t not in expected_types})
            total_recommendations = len(recommendations)
            selected_material_count = len(selected)
            generated_file_count = len(generated_files)

            llm_score, llm_reason = self._llm_judge_evaluation(
                scenario=scenario,
                recommendations=recommendations,
                coverage=type_coverage,
                high_priority_coverage=high_priority_coverage,
                generated_file_count=generated_file_count,
            )

            result = ScenarioEvaluationResult(
                scenario_id=scenario["scenario_id"],
                description=scenario["description"],
                expected_material_types=expected_types,
                recommended_material_types=recommended_types,
                expected_high_priority_types=expected_high,
                high_priority_matches=high_priority_matches,
                type_coverage=type_coverage,
                high_priority_coverage=high_priority_coverage,
                extra_material_types=extra_types,
                total_recommendations=total_recommendations,
                selected_material_count=selected_material_count,
                generated_file_count=generated_file_count,
                llm_judge_score=llm_score,
                llm_judge_reasoning=llm_reason,
            )
            results.append(result)
            self._save_result_to_db(result)

            total_type_coverage += type_coverage
            total_high_priority_coverage += high_priority_coverage
            total_llm_score += llm_score

            print(f"  → Recommended types: {recommended_types}")
            print(f"  → Type coverage: {type_coverage:.2f}")
            print(f"  → High-priority coverage: {high_priority_coverage:.2f}")
            print(f"  → Generated files: {generated_file_count}")
            print(f"  → LLM judge score: {llm_score:.1f}/100")
            print()

        scenario_count = len(scenarios) or 1
        summary = {
            "average_type_coverage": total_type_coverage / scenario_count,
            "average_high_priority_coverage": total_high_priority_coverage / scenario_count,
            "average_llm_judge_score": total_llm_score / scenario_count,
            "scenarios_evaluated": scenario_count,
        }
        print(f"Summary metrics: {json.dumps(summary, indent=2)}")
        return summary

    # LLM judge --------------------------------------------------------
    def _llm_judge_evaluation(
        self,
        scenario: ScenarioConfig,
        recommendations: List[Dict],
        coverage: float,
        high_priority_coverage: float,
        generated_file_count: int,
    ) -> Tuple[float, str]:
        rec_summaries = []
        for rec in recommendations:
            rec_summaries.append(
                f"- {rec.get('material_type', 'unknown')} | priority={rec.get('priority')} | title={rec.get('title')}"
            )
        rec_text = "\n".join(rec_summaries) if rec_summaries else "- (no recommendations)"

        prompt = f"""
You are evaluating a marketing materials planning agent. Rate the quality of its recommendations.

Scenario description: {scenario['description']}
Client context: {scenario['client_context']}

Expected material types: {scenario['expected_material_types']}
Expected high-priority types: {scenario['expected_high_priority_types']}

Agent produced the following recommendations:
{rec_text}

Additional metrics:
- Type coverage (expected vs. recommended): {coverage:.2f}
- High-priority coverage: {high_priority_coverage:.2f}
- Generated asset count: {generated_file_count}

Scoring rubric (0-100):
1. Coverage of critical materials (50 points)
2. Priority alignment and usefulness (30 points)
3. Overall clarity/variety of recommendations (20 points)

Respond with:
SCORE: <number between 0 and 100>
REASONING: <concise justification>
"""

        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            score = 0.0
            reasoning = ""
            for line in text.splitlines():
                if line.upper().startswith("SCORE:"):
                    try:
                        score = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        continue
                elif line.upper().startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip()
                elif reasoning:
                    reasoning += " " + line.strip()
            score = max(0.0, min(100.0, score))
            if not reasoning:
                reasoning = text
            return score, reasoning
        except Exception as exc:  # pragma: no cover - defensive
            return 0.0, f"Error during LLM evaluation: {exc}"


def main() -> None:
    dataset_path = os.path.join(os.path.dirname(__file__), "dataset.json")
    db_path = os.path.join(os.path.dirname(__file__), "evaluation_results.db")

    # Fresh evaluation each run
    if os.path.exists(db_path):
        os.remove(db_path)

    evaluator = MaterialsAgentEvaluator(db_path=db_path)
    evaluator.evaluate(dataset_path)


if __name__ == "__main__":
    main()
