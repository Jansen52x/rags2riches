"""
Convert Fact-Checker Claims to RAG Test Questions

This script converts fact-checking claims into natural questions
suitable for RAG evaluation.
"""

import json
import os
import google.generativeai as genai
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv

def convert_claims_to_questions(
    input_file: str,
    output_file: str,
    google_api_key: str = None
) -> List[Dict]:
    """
    Convert fact-checker claims to natural RAG questions using Gemini

    Args:
        input_file: Path to test_claims.json
        output_file: Path to save converted questions
        google_api_key: Google API key (or uses GOOGLE_API_KEY env var)

    Returns:
        List of converted test cases
    """
    # Load claims
    with open(input_file, 'r') as f:
        data = json.load(f)

    claims = data['claims']

    # Initialize Gemini
    api_key = google_api_key or os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY required")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')

    converted_cases = []

    print(f"Converting {len(claims)} claims to RAG questions...")
    print("=" * 60)

    for i, claim_data in enumerate(claims, 1):
        claim = claim_data['claim']
        expected_verdict = claim_data['expected_verdict']
        claim_id = claim_data['id']

        # Generate natural question from claim
        prompt = f"""Convert this factual claim into a natural question that someone might ask a RAG system.

Claim: "{claim}"

Generate ONE natural question that asks for the information in this claim.

Rules:
- Make it sound like a real user query
- Keep it concise (1 sentence)
- Focus on the key information being claimed
- Output ONLY the question, nothing else

Question:"""

        try:
            response = model.generate_content(prompt)
            question = response.text.strip()

            # Remove quotes if present
            question = question.strip('"\'')

            # Generate expected answer based on verdict
            if expected_verdict == "TRUE":
                expected_answer = f"Yes, {claim}"
            elif expected_verdict == "FALSE":
                expected_answer = f"No, this is incorrect. {claim.replace(' reported ', ' did not report ').replace(' announced ', ' did not announce ')}"
            else:  # CANNOT BE DETERMINED
                expected_answer = "This information cannot be verified from available sources."

            converted_case = {
                "id": claim_id,
                "query": question,
                "original_claim": claim,
                "expected_verdict": expected_verdict,
                "expected_answer_guidance": expected_answer,
                "evaluation_criteria": {
                    "should_retrieve_relevant_docs": expected_verdict != "CANNOT BE DETERMINED",
                    "answer_should_confirm": expected_verdict == "TRUE",
                    "answer_should_deny": expected_verdict == "FALSE",
                    "answer_should_indicate_uncertainty": expected_verdict == "CANNOT BE DETERMINED"
                }
            }

            converted_cases.append(converted_case)

            print(f"[{i}/{len(claims)}] Converted:")
            print(f"  Claim: {claim[:80]}...")
            print(f"  Question: {question}")
            print(f"  Verdict: {expected_verdict}")
            print()

        except Exception as e:
            print(f"[{i}/{len(claims)}] ERROR: {e}")
            # Add placeholder
            converted_cases.append({
                "id": claim_id,
                "query": claim,  # Fallback to using claim as-is
                "original_claim": claim,
                "expected_verdict": expected_verdict,
                "error": str(e)
            })

    # Save to file
    output_data = {
        "metadata": {
            "source": "fact-checker test_claims.json",
            "converted_by": "convert_claims_to_questions.py",
            "total_cases": len(converted_cases)
        },
        "test_cases": converted_cases
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print("=" * 60)
    print(f"✅ Conversion complete!")
    print(f"Saved {len(converted_cases)} test cases to: {output_file}")

    return converted_cases


if __name__ == "__main__":
    load_dotenv()
    # Default paths
    input_file = "../../fact-checker-agent/test_claims.json"
    output_file = "rag_test_questions.json"

    # Check if input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file not found: {input_file}")
        print("Please provide the correct path to test_claims.json")
        exit(1)

    # Convert
    convert_claims_to_questions(input_file, output_file)

    print("\nExample test cases:")
    with open(output_file, 'r') as f:
        data = json.load(f)
        for case in data['test_cases'][:3]:
            print(f"\nQuery: {case['query']}")
            print(f"Expected Verdict: {case['expected_verdict']}")
