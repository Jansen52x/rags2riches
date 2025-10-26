"""
Integration bridge between fact-checker agent and materials decision agent.
This module handles the handoff of verified claims to the materials agent.
"""

import json
import uuid
from typing import List, Dict, Optional
from datetime import datetime
from zoneinfo import ZoneInfo

def extract_verified_claims_from_fact_checker(fact_check_state: Dict) -> List[Dict]:
    """
    Extract and format verified claims from fact-checker output
    
    Args:
        fact_check_state: The final state from fact-checker workflow
        
    Returns:
        List of formatted claims ready for materials agent
    """
    
    verified_claims = []
    claim_verdicts = fact_check_state.get("claim_verdicts", [])
    
    for i, verdict in enumerate(claim_verdicts):
        # Only include claims that passed fact-checking
        if verdict.get("pass_to_materials_agent") == True:
            
            formatted_claim = {
                "claim_id": f"claim_{i+1:03d}",
                "claim": verdict.get("claim", f"Claim {i+1}"),
                "verdict": verdict.get("overall_verdict", "UNKNOWN"),
                "confidence": calculate_confidence_score(verdict),
                "evidence": format_evidence(verdict.get("main_evidence", [])),
                "explanation": verdict.get("explanation", ""),
                "source_fact_check": {
                    "salesperson_id": fact_check_state.get("salesperson_id"),
                    "original_claim": fact_check_state.get("original_claim"),
                    "client_context": fact_check_state.get("client_context"),
                    "timestamp": datetime.now(ZoneInfo("Asia/Singapore")).isoformat()
                }
            }
            
            verified_claims.append(formatted_claim)
    
    return verified_claims

def calculate_confidence_score(verdict: Dict) -> float:
    """
    Calculate confidence score based on verdict quality and evidence
    
    Args:
        verdict: Verdict dictionary from fact-checker
        
    Returns:
        Confidence score between 0.0 and 1.0
    """
    
    base_confidence = 0.5
    
    # Boost confidence for TRUE verdicts
    if verdict.get("overall_verdict") == "TRUE":
        base_confidence += 0.3
    elif verdict.get("overall_verdict") == "FALSE":
        base_confidence += 0.1  # False claims can still be confidently false
    
    # Boost based on evidence quality
    evidence = verdict.get("main_evidence", [])
    if len(evidence) >= 3:
        base_confidence += 0.2
    elif len(evidence) >= 2:
        base_confidence += 0.1
    
    # Check for high-quality sources
    high_quality_sources = ["government", "academic", "official", "research"]
    for evidence_item in evidence:
        source = evidence_item.get("source", "").lower()
        if any(quality_indicator in source for quality_indicator in high_quality_sources):
            base_confidence += 0.1
            break
    
    return min(base_confidence, 1.0)

def format_evidence(evidence_list: List[Dict]) -> List[Dict]:
    """
    Format evidence from fact-checker for materials agent consumption
    
    Args:
        evidence_list: List of evidence dictionaries
        
    Returns:
        Formatted evidence list
    """
    
    formatted_evidence = []
    
    for evidence in evidence_list:
        if isinstance(evidence, dict):
            formatted_evidence.append({
                "source": evidence.get("source", "Unknown source"),
                "summary": evidence.get("summary", "No summary provided"),
                "credibility": assess_source_credibility(evidence.get("source", "")),
                "relevance": "high"  # Assume high relevance since it passed fact-checking
            })
    
    return formatted_evidence

def assess_source_credibility(source: str) -> str:
    """
    Assess the credibility level of a source
    
    Args:
        source: Source name/description
        
    Returns:
        Credibility level: "high", "medium", or "low"
    """
    
    source_lower = source.lower()
    
    high_credibility_indicators = [
        "government", "official", "academic", "university", "research", 
        "institute", "bureau", "ministry", "statistics", "census"
    ]
    
    medium_credibility_indicators = [
        "news", "times", "reuters", "bloomberg", "wsj", "economist",
        "techcrunch", "industry report", "association"
    ]
    
    if any(indicator in source_lower for indicator in high_credibility_indicators):
        return "high"
    elif any(indicator in source_lower for indicator in medium_credibility_indicators):
        return "medium"
    else:
        return "low"

def create_materials_session(verified_claims: List[Dict], 
                           salesperson_id: str,
                           client_context: str) -> Dict:
    """
    Create a materials decision session from verified claims
    
    Args:
        verified_claims: List of verified claims
        salesperson_id: ID of the salesperson
        client_context: Context about the client
        
    Returns:
        Materials session dictionary
    """
    
    from materials_decision_agent import MaterialsDecisionState
    
    session_state = MaterialsDecisionState(
        session_id=str(uuid.uuid4()),
        salesperson_id=salesperson_id,
        client_context=client_context,
        verified_claims=verified_claims,
        material_recommendations=[],
        selected_materials=[],
        generation_queue=[],
        decision_complete=False,
        user_feedback=None
    )
    
    return session_state

def run_complete_pipeline(fact_check_state: Dict) -> Dict:
    """
    Run the complete pipeline from fact-checker output to materials recommendations
    
    Args:
        fact_check_state: Final state from fact-checker workflow
        
    Returns:
        Final materials decision state
    """
    
    # Extract verified claims
    verified_claims = extract_verified_claims_from_fact_checker(fact_check_state)
    
    if not verified_claims:
        return {
            "status": "no_verified_claims",
            "message": "No claims passed fact-checking for materials generation"
        }
    
    # Create materials session
    session_state = create_materials_session(
        verified_claims=verified_claims,
        salesperson_id=fact_check_state.get("salesperson_id", "unknown"),
        client_context=fact_check_state.get("client_context", "")
    )
    
    # Run materials decision workflow
    try:
        from materials_decision_agent import create_materials_decision_workflow
        
        workflow = create_materials_decision_workflow()
        final_state = workflow.invoke(session_state)
        
        return {
            "status": "success",
            "session_id": final_state["session_id"],
            "verified_claims_count": len(verified_claims),
            "recommendations_count": len(final_state["material_recommendations"]),
            "selected_materials_count": len(final_state["selected_materials"]),
            "final_state": final_state
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error running materials workflow: {str(e)}"
        }

# Example usage and testing
if __name__ == "__main__":
    # Example fact-checker output
    sample_fact_check_state = {
        "claim_id": "test_claim_001",
        "original_claim": "Singapore's e-commerce market reached SGD 9 billion in 2023",
        "salesperson_id": "SP12345",
        "client_context": "Small e-commerce startup in Singapore",
        "claim_verdicts": [
            {
                "overall_verdict": "TRUE",
                "explanation": "Official government statistics confirm this figure",
                "main_evidence": [
                    {
                        "source": "Singapore Department of Statistics",
                        "summary": "E-commerce sales reached SGD 9.1 billion in 2023"
                    },
                    {
                        "source": "Economic Development Board Singapore",
                        "summary": "Strong growth in digital commerce sector"
                    }
                ],
                "pass_to_materials_agent": True
            }
        ]
    }
    
    # Test the integration
    print("Testing Fact-Checker to Materials Agent Integration")
    print("=" * 60)
    
    # Extract verified claims
    verified_claims = extract_verified_claims_from_fact_checker(sample_fact_check_state)
    print(f"Extracted {len(verified_claims)} verified claims")
    
    for claim in verified_claims:
        print(f"\nClaim: {claim['claim']}")
        print(f"Verdict: {claim['verdict']}")
        print(f"Confidence: {claim['confidence']:.2f}")
        print(f"Evidence sources: {len(claim['evidence'])}")
    
    # Test complete pipeline
    print("\nRunning complete pipeline...")
    result = run_complete_pipeline(sample_fact_check_state)
    print(f"Pipeline result: {result['status']}")
    
    if result['status'] == 'success':
        print(f"Verified claims processed: {result['verified_claims_count']}")
        print(f"Materials recommended: {result['recommendations_count']}")
        print(f"Materials selected: {result['selected_materials_count']}")