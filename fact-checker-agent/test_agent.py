"""
Fact Checker Agent Evaluation Script

This script evaluates a fact checker agent using two methods:
1. Exact match accuracy - comparing verdicts directly
2. LLM-judged quality score - using Gemini to evaluate reasoning quality

Requirements:
    pip install google-generativeai
    
Environment variable needed:
    GEMINI_API_KEY - Your Google AI Studio API key
"""

import json
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass
import google.generativeai as genai


@dataclass
class FactCheckResult:
    """Result from the fact checker agent"""
    claim: str
    verdict: str  # TRUE/FALSE/CANNOT BE DETERMINED
    explanation: str


@dataclass
class EvaluationResult:
    """Evaluation results for a single claim"""
    claim_id: str
    claim: str
    expected_verdict: str
    actual_verdict: str
    exact_match: bool
    llm_judge_score: float  # 0-100
    llm_judge_reasoning: str


class FactCheckerEvaluator:
    """Evaluates fact checker agent performance"""
    
    def __init__(self, gemini_api_key: str = None):
        """
        Initialize evaluator with Gemini API
        
        Args:
            gemini_api_key: Google AI API key (or set GEMINI_API_KEY env var)
        """
        api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("Gemini API key required. Set GEMINI_API_KEY env variable or pass as parameter.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def load_test_data(self, filepath: str) -> List[Dict]:
        """Load test claims from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data['claims']
    
    def exact_match_evaluation(self, expected: str, actual: str) -> bool:
        """
        Simple exact match evaluation
        
        Args:
            expected: Expected verdict (TRUE/FALSE/CANNOT BE DETERMINED)
            actual: Actual verdict from fact checker
            
        Returns:
            True if verdicts match exactly
        """
        return expected.upper().strip() == actual.upper().strip()
    
    def llm_judge_evaluation(self, claim: str, expected_verdict: str, 
                            actual_verdict: str, actual_explanation: str) -> Tuple[float, str]:
        """
        Use Gemini as an LLM judge to evaluate fact checker quality
        
        Args:
            claim: The original claim being fact-checked
            expected_verdict: Expected verdict
            actual_verdict: Actual verdict from agent
            actual_explanation: Actual reasoning from agent
            
        Returns:
            Tuple of (score 0-100, reasoning)
        """
        
        prompt = f"""You are evaluating a fact-checking agent's performance. 

CLAIM: {claim}

EXPECTED VERDICT: {expected_verdict}

ACTUAL VERDICT: {actual_verdict}
ACTUAL EXPLANATION: {actual_explanation}

Evaluate the fact checker's response on a scale of 0-100 based on:
1. Verdict correctness (50 points): Is the verdict correct?
2. Reasoning quality (40 points): Is the explanation logical, accurate, and well-supported?
3. Clarity (10 points): Is the explanation clear and easy to understand?

Scoring guidelines:
- If the verdict is completely wrong: Maximum 40 points total (can still get points for reasoning quality)
- If the verdict is correct but explanation is poor or wrong: 50-65 points
- If the verdict is correct with decent explanation: 65-85 points
- If the verdict is correct with excellent explanation: 85-100 points

Provide your response in exactly this format:
SCORE: [number between 0-100]
REASONING: [Your detailed reasoning for the score]
"""
        
        try:
            response = self.model.generate_content(prompt)
            result_text = response.text
            
            # Parse score and reasoning
            lines = result_text.strip().split('\n')
            score = None
            reasoning_lines = []
            
            for line in lines:
                if line.startswith('SCORE:'):
                    score_str = line.replace('SCORE:', '').strip()
                    score = float(score_str)
                elif line.startswith('REASONING:'):
                    reasoning_lines.append(line.replace('REASONING:', '').strip())
                elif reasoning_lines:  # Continue collecting reasoning
                    reasoning_lines.append(line.strip())
            
            reasoning = ' '.join(reasoning_lines)
            
            if score is None:
                # Fallback parsing
                import re
                score_match = re.search(r'(\d+(?:\.\d+)?)', result_text)
                score = float(score_match.group(1)) if score_match else 0.0
                reasoning = result_text
            
            # Clamp score between 0-100
            score = max(0.0, min(100.0, score))
            
            return score, reasoning
            
        except Exception as e:
            print(f"Error in LLM judge evaluation: {e}")
            return 0.0, f"Error during evaluation: {str(e)}"
    
    def evaluate(self, test_data_path: str, 
                fact_checker_func) -> Dict:
        """
        Run full evaluation on the fact checker agent
        
        Args:
            test_data_path: Path to test data JSON file
            fact_checker_func: Function that takes a claim string and returns 
                             FactCheckResult(claim, verdict, explanation)
        
        Returns:
            Dictionary with evaluation metrics and detailed results
        """
        test_claims = self.load_test_data(test_data_path)
        results = []
        
        exact_matches = 0
        total_llm_score = 0.0
        
        print(f"\n{'='*80}")
        print(f"FACT CHECKER EVALUATION - {len(test_claims)} claims")
        print(f"{'='*80}\n")
        
        for claim_data in test_claims:
            claim_id = claim_data['id']
            claim = claim_data['claim']
            expected_verdict = claim_data['expected_verdict']
            
            print(f"Testing [Claim {claim_id}]")
            print(f"Claim: {claim}")
            
            # Run fact checker
            result = fact_checker_func(claim)
            
            # Exact match evaluation
            exact_match = self.exact_match_evaluation(expected_verdict, result.verdict)
            if exact_match:
                exact_matches += 1
            
            # LLM judge evaluation
            llm_score, llm_reasoning = self.llm_judge_evaluation(
                claim, expected_verdict,
                result.verdict, result.explanation
            )
            total_llm_score += llm_score
            
            # Store result
            eval_result = EvaluationResult(
                claim_id=claim_id,
                claim=claim,
                expected_verdict=expected_verdict,
                actual_verdict=result.verdict,
                exact_match=exact_match,
                llm_judge_score=llm_score,
                llm_judge_reasoning=llm_reasoning
            )
            results.append(eval_result)
            
            # Print immediate feedback
            match_symbol = "✓" if exact_match else "✗"
            print(f"  Expected: {expected_verdict} | Actual: {result.verdict} [{match_symbol}]")
            print(f"  LLM Judge Score: {llm_score:.1f}/100")
            print(f"  Agent Explanation: {result.explanation[:100]}...")
            print()
        
        # Calculate overall metrics
        exact_match_accuracy = (exact_matches / len(test_claims)) * 100
        avg_llm_score = total_llm_score / len(test_claims)
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(f"Total Claims Tested: {len(test_claims)}")
        print(f"\n1. EXACT MATCH ACCURACY: {exact_match_accuracy:.1f}%")
        print(f"   Correct: {exact_matches}/{len(test_claims)}")
        
        print(f"\n2. LLM JUDGE QUALITY SCORE: {avg_llm_score:.1f}/100")
        print(f"   Average quality rating across all claims")
        
        return {
            'exact_match_accuracy': exact_match_accuracy,
            'exact_matches': exact_matches,
            'total_claims': len(test_claims),
            'avg_llm_judge_score': avg_llm_score,
            'detailed_results': results
        }
    
    def save_detailed_report(self, evaluation_results: Dict, output_path: str):
        """Save detailed evaluation report to JSON file"""
        
        # Convert results to serializable format
        report = {
            'summary': {
                'exact_match_accuracy': evaluation_results['exact_match_accuracy'],
                'exact_matches': evaluation_results['exact_matches'],
                'total_claims': evaluation_results['total_claims'],
                'avg_llm_judge_score': evaluation_results['avg_llm_judge_score']
            },
            'detailed_results': [
                {
                    'claim_id': r.claim_id,
                    'claim': r.claim,
                    'expected_verdict': r.expected_verdict,
                    'actual_verdict': r.actual_verdict,
                    'exact_match': r.exact_match,
                    'llm_judge_score': r.llm_judge_score,
                    'llm_judge_reasoning': r.llm_judge_reasoning
                }
                for r in evaluation_results['detailed_results']
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDetailed report saved to: {output_path}")


# Example usage and mock fact checker for testing
def mock_fact_checker(claim: str) -> FactCheckResult:
    """
    Mock fact checker for demonstration purposes.
    Replace this with your actual fact checker agent.
    """
    # This is a placeholder - replace with your actual agent
    import random
    
    verdicts = ["TRUE", "FALSE", "CANNOT BE DETERMINED"]
    verdict = random.choice(verdicts)
    explanation = f"This claim is {verdict.lower()} because [mock reasoning]."
    
    return FactCheckResult(claim, verdict, explanation)


if __name__ == "__main__":
    # Initialize evaluator
    # Make sure to set GEMINI_API_KEY environment variable
    # export GEMINI_API_KEY="your-api-key-here"
    
    try:
        evaluator = FactCheckerEvaluator()
        
        # Run evaluation
        # Replace mock_fact_checker with your actual fact checker function
        results = evaluator.evaluate(
            test_data_path='test_claims.json',
            fact_checker_func=mock_fact_checker
        )
        
        # Save detailed report
        evaluator.save_detailed_report(results, 'evaluation_report.json')
        
    except ValueError as e:
        print(f"\nError: {e}")
        print("\nTo use this script:")
        print("1. Get a Gemini API key from https://aistudio.google.com/app/apikey")
        print("2. Set it as an environment variable: export GEMINI_API_KEY='your-key'")
        print("3. Run the script again")