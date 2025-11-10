"""
Materials Decision UI - Streamlit-based interface for salespeople
to review and approve presentation materials recommendations
"""

import streamlit as st
import json
import uuid
from datetime import datetime
import requests
import os

print("[Marketing] Session keys:", list(st.session_state.keys()))
print("[Marketing] materials_verified_claims:", st.session_state.get("materials_verified_claims"))
# FastAPI endpoints
FASTAPI_INTERNAL_URL = os.getenv("FASTAPI_INTERNAL_URL", "http://fastapi_service:8001")
FASTAPI_PUBLIC_URL = os.getenv("FASTAPI_PUBLIC_URL", "http://localhost:8001")

# st.session_state:
# claim - original claim that was verified
# client_context - context about the client that the sales agent is meeting

# Streamlit page config
st.set_page_config(
    page_title="Marketing Materials Decision Service",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #1f77b4, #ff7f0e);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}

.material-card {
    border: 1px solid #dcdcdc;
    border-radius: 10px;
    padding: 1.2rem;
    margin: 0.75rem 0;
    background: #ffffff; /* ensure readable on dark theme */
    color: #111;
}

.material-card h4,
.material-card p,
.material-card li,
.material-card strong,
.material-card span {
    color: #111 !important; /* override Streamlit dark text color inside white card */
}

.priority-high {
    border-left: 5px solid #ff4444;
}

.priority-medium {
    border-left: 5px solid #ffaa00;
}

.priority-low {
    border-left: 5px solid #44ff44;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []
if 'selected_materials' not in st.session_state:
    st.session_state.selected_materials = []
if 'workflow_complete' not in st.session_state:
    st.session_state.workflow_complete = False
if 'verified_claims' not in st.session_state:
    st.session_state.verified_claims = []
if 'generated_files' not in st.session_state:
    st.session_state.generated_files = []

# Sync verified claims from fact checker session if present
if st.session_state.get("materials_verified_claims"):
    st.session_state.verified_claims = st.session_state.materials_verified_claims
if 'verified_claims' not in st.session_state:
    st.session_state.verified_claims = []

# Check if claim from Fact Checker exists and add it to verified_claims
if ('claim' in st.session_state and 
    'claim_verdict' in st.session_state and 
    st.session_state.claim and
    st.session_state.claim_verdict):
    
    # Transform to verified_claims format
    fact_checker_claim = {
        "claim_text": st.session_state.claim,
        "verdict": st.session_state.claim_verdict.get('overall_verdict', 'UNKNOWN'),
        "confidence": 0.85,  # Default confidence
        "evidence": st.session_state.claim_verdict.get('main_evidence', []),
        "explanation": st.session_state.claim_verdict.get('explanation', ''),
        "pass_to_materials_agent": st.session_state.claim_verdict.get('pass_to_materials_agent', False)
    }
    
    # Add to verified_claims if not already present (avoid duplicates)
    claim_texts = [c.get('claim_text', '') for c in st.session_state.verified_claims]
    if fact_checker_claim['claim_text'] not in claim_texts:
        st.session_state.verified_claims.append(fact_checker_claim)
        
        # Clear the fact checker session state to prevent re-adding
        st.session_state.claim = ""
        st.session_state.claim_verdict = None

# Header
st.markdown(f"""
<div class="main-header">
    <h1>Meeting Materials Decision Agent</h1>
    <p>Presentation materials recommendation and management</p>
</div>
""", unsafe_allow_html=True)


def _normalize_public_path(path: str) -> str:
    """Ensure the generated asset path works with FASTAPI_PUBLIC_URL."""
    if not path:
        return path
    if not path.startswith("/"):
        return f"/{path}"
    return path


def render_generated_assets(generated: list[str], header_text="🖼️ Generated Materials") -> None:
    """Render generated media assets in a two-column layout."""
    if not generated:
        return
    st.header(header_text)
    cols = st.columns(2)
    for idx, file_path in enumerate(generated):
        col = cols[idx % 2]
        with col:
            public_path = _normalize_public_path(file_path)
            full_url = f"{FASTAPI_PUBLIC_URL}{public_path}"
            lower = public_path.lower()
            if lower.endswith((".png", ".jpg", ".jpeg")):
                st.image(full_url)
            elif lower.endswith(".mp4"):
                st.video(full_url)


if st.session_state.get("generated_files"):
    render_generated_assets(st.session_state.generated_files)

# Sidebar for inputs
with st.sidebar:
    st.header("📋 Session Setup")
    
    salesperson_id = st.text_input(
        "Salesperson ID", 
        value="SP12345",
        help="Your unique salesperson identifier"
    )
    
    client_name = st.text_input(
        "Client Name",
        value="TechCorp Inc",
        help="Name of the client company"
    )
    
    meeting_type = st.selectbox(
        "Meeting Type",
        ["initial", "follow_up", "proposal", "presentation"],
        help="Type of meeting this material is for"
    )
    
    creative_prompt = st.text_input(
        "Creative prompt (optional)",
        value="",
        help="Extra guidance for image/video style, e.g. 'cafe theme, warm tones, natural light'"
    )
    
    st.header("Verified Claims Input")
    
    # Show claims from Fact Checker if they exist
    fact_checker_claims = [
        c for c in st.session_state.get('verified_claims', []) if c.get('verdict')
    ]
    if fact_checker_claims:
        st.success(f"✓ {len(fact_checker_claims)} claim(s) received from Fact Checker")
        for claim in fact_checker_claims:
            claim_text = claim.get('claim') or claim.get('claim_text') or "Unknown claim"
            with st.expander(f"🔍 {claim_text[:50]}..."):
                st.write(f"**Verdict:** {claim.get('verdict', 'UNKNOWN')}")
                confidence = claim.get('confidence')
                if isinstance(confidence, (float, int)):
                    st.write(f"**Confidence:** {confidence:.0%}")
                if claim.get('explanation'):
                    st.write(f"**Explanation:** {claim['explanation']}")
    
    # Sample verified claims for testing
    sample_claims = [
        {
            "claim_text": "Our company has 30% market share in Singapore",
            "confidence": 0.9
        },
        {
            "claim_text": "We have expanded to 5 Southeast Asian countries",
            "confidence": 0.85
        }
    ]
    
    if st.button("Load Sample Claims"):
        st.session_state.verified_claims = sample_claims
        st.success("Sample claims loaded!")

# Main content area
verified_claims = st.session_state.get('verified_claims', [])

if verified_claims:
    st.header("Current Claims")
    for i, claim in enumerate(verified_claims):
        claim_text = claim.get('claim') or claim.get('claim_text') or "Unknown claim"
        confidence = claim.get('confidence')
        with st.expander(f"Claim {i+1}: {claim_text[:60]}..."):
            if isinstance(confidence, (float, int)):
                st.write(f"**Confidence:** {confidence:.0%}")
            verdict = claim.get('verdict') or claim.get('overall_verdict')
            if verdict:
                st.write(f"**Verdict:** {verdict}")
            evidence = claim.get('evidence') or claim.get('main_evidence') or []
            if evidence:
                st.write("**Evidence:**")
                for ev in evidence:
                    source = ev.get('source', 'Unknown source') if isinstance(ev, dict) else str(ev)
                    summary = ev.get('summary', '') if isinstance(ev, dict) else ''
                    bullet = f"- {source}"
                    if summary:
                        bullet += f": {summary}"
                    st.write(bullet)
    
    # Generate materials button
    if st.button("🎨 Generate Marketing Materials", type="primary"):
        with st.spinner("Analyzing claims and generating materials..."):
            try:
                st.info("Attempting to connect to FastAPI service...")
                # Call FastAPI endpoint
                normalized_claims = []
                for idx, claim in enumerate(verified_claims, start=1):
                    if not isinstance(claim, dict):
                        continue

                    claim_text = claim.get('claim') or claim.get('claim_text')
                    if not claim_text:
                        continue

                    normalized_claims.append({
                        "claim_id": claim.get('claim_id') or f"claim_{idx:03d}",
                        "claim": claim_text,
                        "verdict": claim.get('verdict') or claim.get('overall_verdict', 'TRUE'),
                        "confidence": claim.get('confidence', 0.75),
                        "evidence": claim.get('evidence') or claim.get('main_evidence', []),
                        "explanation": claim.get('explanation', ''),
                        "pass_to_materials_agent": claim.get('pass_to_materials_agent', True),
                    })

                if not normalized_claims:
                    st.error("No valid verified claims available for materials generation.")
                else:
                    payload = {
                        "verified_claims": normalized_claims,
                        "salesperson_id": salesperson_id,
                        "client_context": json.dumps({
                            "client_name": client_name,
                            "meeting_type": meeting_type
                        }),
                        "user_prompt": creative_prompt or None
                    }
                    
                    # Debug information
                    st.write("Making request to FastAPI service...")
                    st.write(f"Payload: {json.dumps(payload, indent=2)}")
                    
                    response = requests.post(
                        f"{FASTAPI_INTERNAL_URL}/generate-materials",
                        json=payload,
                        timeout=3000  # Increased timeout for real generation
                    )
                
                    if response.ok:
                        result = response.json()
                        st.success("✅ Materials generation complete!")
                        
                        # Store recommendations and update UI
                        st.session_state.recommendations = result.get("recommendations", [])
                        st.session_state.selected_materials = result.get("selected_materials", [])
                        st.session_state.workflow_complete = True
                        st.session_state.generated_files = result.get("generated_files", [])
                        
                        render_generated_assets(st.session_state.generated_files)

                        # Show recommendations
                        st.header("📋 Material Recommendations")
                        for i, rec in enumerate(result.get("recommendations", [])):
                            priority_class = f"priority-{rec.get('priority', 'medium')}"

                            st.markdown(f"""
                            <div class="material-card {priority_class}">
                                <h4>{rec['title']}</h4>
                                <p><strong>Type:</strong> {rec['material_type'].replace('_', ' ').title()}</p>
                                <p><strong>Priority:</strong> {rec['priority'].upper()}</p>
                                <p><strong>Estimated Time:</strong> {rec['estimated_time_minutes']} minutes</p>
                                <p><strong>Description:</strong> {rec['description']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.error(f"Error response from FastAPI service: {response.status_code}")
                        st.error(f"Response text: {response.text}")
                    
            except requests.exceptions.ConnectionError as e:
                st.error("Could not connect to FastAPI service. Make sure the service is running.")
                st.error(f"Error details: {str(e)}")
            except requests.exceptions.Timeout as e:
                st.error("Request to FastAPI service timed out.")
                st.error(f"Error details: {str(e)}")
            except Exception as e:
                st.error(f"Error generating materials: {str(e)}")
                st.error("Exception type: " + str(type(e)))

# Refresh generated materials section
st.markdown("---")
if st.button("🔄 Refresh Generated Materials"):
    try:
        response = requests.get(f"{FASTAPI_INTERNAL_URL}/generated-files")
        if response.ok:
            files = response.json().get("files", [])
            if files:
                st.session_state.generated_files = [file["url"] for file in files]
                render_generated_assets(st.session_state.generated_files, header_text="🖼️ Current Generated Materials")
            else:
                st.info("No generated materials found")
    except Exception as e:
        st.error(f"Error refreshing materials: {str(e)}")

# Reset session button
if st.sidebar.button("Reset Session"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.session_id = str(uuid.uuid4())
    st.rerun()