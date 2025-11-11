"""
Batch RAG Evaluation Script

Run RAG system evaluation on a test dataset and generate aggregate metrics.
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime
import logging
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_services import RAGService, EmbeddingService, LLMService, RAGEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchRAGEvaluator:
    """Batch evaluation for RAG system on test datasets"""

    def __init__(self, rag_service: RAGService, use_non_llm_metrics: bool = True):
        """
        Initialize batch evaluator

        Args:
            rag_service: RAG service instance
            use_non_llm_metrics: Whether to compute non-LLM metrics
        """
        self.rag_service = rag_service
        self.use_non_llm_metrics = use_non_llm_metrics
        if use_non_llm_metrics:
            self.non_llm_evaluator = RAGEvaluator()

    def load_test_dataset(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load test dataset from JSON file

        Expected format:
        {
          "test_cases": [
            {
              "id": 1,
              "query": "What was Salesforce's revenue growth?",
              "expected_info": "...",
              ...
            }
          ]
        }
        """
        with open(file_path, 'r') as f:
            data = json.load(f)

        return data.get('test_cases', [])

    def evaluate_dataset(
        self,
        test_cases: List[Dict[str, Any]],
        output_file: str = None,
        evaluate_with_llm_judge: bool = True
    ) -> Dict[str, Any]:
        """
        Run RAG system on all test cases and collect metrics

        Args:
            test_cases: List of test case dictionaries
            output_file: Optional path to save detailed results
            evaluate_with_llm_judge: Whether to use LLM-as-judge evaluation

        Returns:
            Dictionary with aggregate metrics and detailed results
        """
        results = []
        total = len(test_cases)

        print(f"\n{'='*80}")
        print(f"BATCH RAG EVALUATION - {total} test cases")
        print(f"{'='*80}\n")

        for i, test_case in enumerate(test_cases, 1):
            query = test_case['query']
            test_id = test_case.get('id', i)

            print(f"[{i}/{total}] Evaluating: {query[:70]}...")

            try:
                # Run RAG query
                rag_result = self.rag_service.query(
                    query,
                    include_sources=True
                )

                answer = rag_result['answer']
                sources = rag_result.get('sources', [])

                # Get context for evaluation
                context = "\n\n".join([
                    f"Source {j+1}:\n{source['content']}"
                    for j, source in enumerate(sources)
                ])

                # LLM-as-judge evaluation (from generate method)
                llm_eval = None
                if evaluate_with_llm_judge and self.rag_service.llm_service.judge_model:
                    try:
                        llm_eval = self.rag_service.llm_service.evaluate_rag_response(
                            query, context, answer, save_to_csv=False
                        )
                    except Exception as e:
                        logger.error(f"LLM evaluation failed: {e}")

                # Non-LLM metrics
                non_llm_metrics = None
                if self.use_non_llm_metrics:
                    try:
                        non_llm_metrics = self.non_llm_evaluator.evaluate_all(
                            query, context, answer, sources
                        )
                        
                        # DEBUG: Print raw metrics for first few examples
                        if i <= 3:
                            print(f"  🔍 DEBUG - Raw metrics for test case {test_id}:")
                            if non_llm_metrics:
                                print(f"    Context utilization: {non_llm_metrics.get('context_utilization', {})}")
                                print(f"    Retrieval metrics: {non_llm_metrics.get('retrieval_metrics', {})}")
                                print(f"    Specificity: {non_llm_metrics.get('specificity_score', {})}")
                    except Exception as e:
                        logger.error(f"Non-LLM metrics failed: {e}")

                # Store result
                result = {
                    "test_id": test_id,
                    "query": query,
                    "answer": answer,
                    "num_sources_retrieved": len(sources),
                    "expected_info": test_case.get('expected_info'),
                    "query_type": test_case.get('query_type'),
                    "difficulty": test_case.get('difficulty'),
                    "companies_involved": test_case.get('companies_involved'),
                    "llm_evaluation": llm_eval,
                    "non_llm_metrics": non_llm_metrics,
                    "sources": [
                        {
                            "content_preview": source['content'][:200],
                            "score": source.get('score', 0)
                        }
                        for source in sources[:3]  # Top 3 sources
                    ]
                }

                results.append(result)

                # Print quick summary
                if llm_eval and 'llm_judge' in llm_eval:
                    llm_judge = llm_eval['llm_judge']
                    if 'faithfulness' in llm_judge:
                        print(f"  Faithfulness: {llm_judge['faithfulness']['score']}/5")
                        print(f"  Answer Relevance: {llm_judge['answer_relevance']['score']}/5")
                if non_llm_metrics:
                    ctx_util = non_llm_metrics.get('context_utilization', {}).get('context_token_usage', 0)
                    # Handle both decimal and percentage formats
                    if ctx_util > 1:
                        print(f"  Context Utilization: {ctx_util:.1f}%")
                    else:
                        print(f"  Context Utilization: {ctx_util:.1%}")

            except Exception as e:
                logger.error(f"Error processing test case {test_id}: {e}")
                results.append({
                    "test_id": test_id,
                    "query": query,
                    "error": str(e)
                })

            print()

        # Calculate aggregate metrics
        aggregate = self._calculate_aggregate_metrics(results)

        # Print summary
        self._print_summary(aggregate)

        # Save detailed results
        if output_file:
            self._save_results(results, aggregate, output_file)

        return {
            "aggregate_metrics": aggregate,
            "detailed_results": results
        }

    def _calculate_aggregate_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregate statistics from results"""

        # Filter out errors
        valid_results = [r for r in results if 'error' not in r]

        if not valid_results:
            return {"error": "No valid results to aggregate"}

        # LLM evaluation aggregates
        llm_scores = {
            "faithfulness": [],
            "answer_relevance": [],
            "context_relevance": []
        }

        for result in valid_results:
            llm_eval = result.get('llm_evaluation')
            if llm_eval and 'llm_judge' in llm_eval:
                llm_judge = llm_eval['llm_judge']
                # Check if evaluation succeeded (no error)
                if 'faithfulness' in llm_judge and 'error' not in llm_judge:
                    llm_scores['faithfulness'].append(llm_judge['faithfulness']['score'])
                    llm_scores['answer_relevance'].append(llm_judge['answer_relevance']['score'])
                    llm_scores['context_relevance'].append(llm_judge['context_relevance']['score'])

        # Non-LLM metric aggregates
        non_llm_aggregates = {
            "avg_word_count": 0,
            "avg_context_utilization": 0,
            "avg_retrieval_score": 0,
            "avg_specificity": 0
        }

        counts = {"word_count": 0, "context_util": 0, "retrieval": 0, "specificity": 0}

        for result in valid_results:
            metrics = result.get('non_llm_metrics')
            if metrics:
                if 'answer_length' in metrics:
                    non_llm_aggregates['avg_word_count'] += metrics['answer_length']['word_count']
                    counts['word_count'] += 1
                if 'context_utilization' in metrics:
                    non_llm_aggregates['avg_context_utilization'] += metrics['context_utilization']['context_token_usage']
                    counts['context_util'] += 1
                if 'retrieval_metrics' in metrics and metrics['retrieval_metrics']:
                    non_llm_aggregates['avg_retrieval_score'] += metrics['retrieval_metrics'].get('avg_retrieval_score', 0)
                    counts['retrieval'] += 1
                if 'specificity_score' in metrics:
                    non_llm_aggregates['avg_specificity'] += metrics['specificity_score']['specificity_score']
                    counts['specificity'] += 1

        # Calculate averages
        for key in non_llm_aggregates:
            count_key = key.replace('avg_', '')
            if counts.get(count_key, 0) > 0:
                non_llm_aggregates[key] /= counts[count_key]

        return {
            "total_cases": len(results),
            "successful_cases": len(valid_results),
            "failed_cases": len(results) - len(valid_results),
            "llm_judge_metrics": {
                "avg_faithfulness": sum(llm_scores['faithfulness']) / len(llm_scores['faithfulness']) if llm_scores['faithfulness'] else None,
                "avg_answer_relevance": sum(llm_scores['answer_relevance']) / len(llm_scores['answer_relevance']) if llm_scores['answer_relevance'] else None,
                "avg_context_relevance": sum(llm_scores['context_relevance']) / len(llm_scores['context_relevance']) if llm_scores['context_relevance'] else None,
                "num_evaluated": len(llm_scores['faithfulness'])
            },
            "non_llm_metrics": non_llm_aggregates
        }

    def _print_summary(self, aggregate: Dict[str, Any]):
        """Print aggregate metrics summary"""
        print(f"\n{'='*80}")
        print("EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(f"Total Test Cases: {aggregate['total_cases']}")
        print(f"Successful: {aggregate['successful_cases']}")
        print(f"Failed: {aggregate['failed_cases']}")

        llm = aggregate.get('llm_judge_metrics', {})
        if llm.get('num_evaluated', 0) > 0:
            print(f"\n📊 LLM-as-Judge Metrics (1-5 scale):")
            print(f"  Average Faithfulness: {llm['avg_faithfulness']:.2f}/5")
            print(f"  Average Answer Relevance: {llm['avg_answer_relevance']:.2f}/5")
            print(f"  Average Context Relevance: {llm['avg_context_relevance']:.2f}/5")

        non_llm = aggregate.get('non_llm_metrics', {})
        if non_llm:
            print(f"\n📈 Non-LLM Metrics:")
            print(f"  Average Answer Length: {non_llm['avg_word_count']:.0f} words")
            
            # Handle context utilization - check if it's already a percentage or decimal
            context_util = non_llm['avg_context_utilization']
            if context_util > 1:  # Already a percentage value
                print(f"  Average Context Utilization: {context_util:.1f}%")
            else:  # Decimal that needs conversion to percentage
                print(f"  Average Context Utilization: {context_util:.1%}")
            
            print(f"  Average Retrieval Score: {non_llm['avg_retrieval_score']:.3f}")
            
            # Handle specificity - check if it's already a percentage or decimal
            specificity = non_llm['avg_specificity']
            if specificity > 1:  # Already a percentage value
                print(f"  Average Specificity: {specificity:.1f}%")
            else:  # Decimal that needs conversion to percentage
                print(f"  Average Specificity: {specificity:.1%}")

        print(f"\n{'='*80}")

    def _save_results(self, results: List[Dict], aggregate: Dict, output_file: str):
        """Save detailed results to JSON file"""
        output_data = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "aggregate_metrics": aggregate,
            "detailed_results": results
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✅ Detailed results saved to: {output_file}")

        # Also save as CSV for easy analysis
        csv_file = output_file.replace('.json', '.csv')
        self._save_results_csv(results, csv_file)

    def _save_results_csv(self, results: List[Dict], csv_file: str):
        """Save results as CSV for spreadsheet analysis"""
        rows = []

        for result in results:
            if 'error' in result:
                continue

            row = {
                'test_id': result['test_id'],
                'query': result['query'],
                'answer': result['answer'][:200],  # Truncate
                'num_sources': result['num_sources_retrieved'],
                'expected_info': result.get('expected_info'),
                'query_type': result.get('query_type'),
                'difficulty': result.get('difficulty'),
            }

            # LLM metrics
            llm_eval = result.get('llm_evaluation')
            if llm_eval and 'llm_judge' in llm_eval:
                llm_judge = llm_eval['llm_judge']
                if 'faithfulness' in llm_judge and 'error' not in llm_judge:
                    row['faithfulness'] = llm_judge['faithfulness']['score']
                    row['answer_relevance'] = llm_judge['answer_relevance']['score']
                    row['context_relevance'] = llm_judge['context_relevance']['score']

            # Non-LLM metrics
            non_llm = result.get('non_llm_metrics')
            if non_llm:
                row['word_count'] = non_llm.get('answer_length', {}).get('word_count')
                row['context_utilization'] = non_llm.get('context_utilization', {}).get('context_token_usage')
                row['specificity'] = non_llm.get('specificity_score', {}).get('specificity_score')
                # Add retrieval score to CSV
                row['retrieval_score'] = non_llm.get('retrieval_metrics', {}).get('avg_retrieval_score')

            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)
        print(f"✅ CSV results saved to: {csv_file}")


def main():
    """Run batch evaluation on test dataset"""

    # Initialize RAG services
    print("Initializing RAG services...")
    embedding_service = EmbeddingService()
    llm_service = LLMService()
    rag_service = RAGService(embedding_service, llm_service)

    # Initialize batch evaluator
    evaluator = BatchRAGEvaluator(rag_service, use_non_llm_metrics=True)

    # Load test dataset
    test_file = "rag_test_questions.json"

    if not Path(test_file).exists():
        print(f"Error: Test file not found: {test_file}")
        print("\nPlease run convert_claims_to_questions.py first to generate the test dataset.")
        return

    test_cases = evaluator.load_test_dataset(test_file)
    print(f"Loaded {len(test_cases)} test cases from {test_file}")

    # Run evaluation
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"evaluation_results_{timestamp}.json"

    results = evaluator.evaluate_dataset(
        test_cases,
        output_file=output_file,
        evaluate_with_llm_judge=True  # Set to False to skip LLM judge
    )


if __name__ == "__main__":
    # Load environment variables FIRST
    load_dotenv()
    main()