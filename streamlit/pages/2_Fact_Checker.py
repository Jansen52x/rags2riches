import streamlit as st
import requests  
import json      
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo 

# Streamlit page config
st.set_page_config(
    page_title="Fact Checker Service",
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

</style>
""", unsafe_allow_html=True)

# --- UI (Good, keep all this) ---
st.markdown(f"""
<div class="main-header">
    <h1>Fact Checking Agent</h1>
    <p>Fact checking the claims from the RAG for your convenience</p>
</div>
""", unsafe_allow_html=True)

if 'workflow_complete' not in st.session_state:
    st.session_state.workflow_complete = False
if 'claim_verdict' not in st.session_state:
    st.session_state.claim_verdict = None
if 'claim' not in st.session_state:
    st.session_state.claim = ""
if 'verifying_claim' not in st.session_state:
    st.session_state.verifying_claim = False
if 'materials_verified_claims' not in st.session_state:
    st.session_state.materials_verified_claims = []
if 'client_context' not in st.session_state:
    st.session_state.client_context = ""

with st.sidebar:
    st.header("📋 Session Setup")
    salesperson_id = st.text_input("Salesperson ID", value="SP12345")

    # Keep client context separate from RAG context
    client_context = st.text_area(
        "Client Context",
        value="Small e-commerce startup in Singapore...",
        height=200,
        help="Background information about the client to help contextualize the fact-check."
    )
    st.session_state.client_context = client_context
    if st.button("Reset Session"):
        st.session_state.clear()
        st.rerun()

# --- Main Content Area ---
if 'claim' in st.session_state:
    st.header("Claim To Verify")

    # Show if this came from RAG
    if 'rag_context' in st.session_state:
        st.info(f"📚 **From RAG Query:** {st.session_state.rag_context}")
        st.markdown("")

    user_claim = st.text_input("Enter the claim you want to verify:", key="claim")

    # ===== THIS IS THE MODIFIED PART =====
    if st.button("Verify claim", type="primary", disabled=st.session_state.verifying_claim):
        
        # --- This is the Sample Request you asked for ---
        # 1. Define the API endpoint
        API_URL = "http://fastapi_service:8001/check-claim"
        
        # 2. Define the JSON payload
        payload = {
            "claim": st.session_state.claim,
            "salesperson_id": salesperson_id,
            "client_context": client_context
        }
        
        # 3. Setup the UI
        progress_bar = st.progress(25)
        progress_text = st.empty()
        progress_text.text("Step 1/4: Analyzing claim...")
        
        final_verdict = None
        st.session_state.verifying_claim = True
        try:
            # 4. Call the API with stream=True
            with requests.post(API_URL, json=payload, stream=True, timeout=300) as response:
                response.raise_for_status() # Raise an error for bad responses
                
                # 5. Iterate over the streaming response line-by-line
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line.decode('utf-8'))
                        # # Check for an error message from the agent
                        type = data.get("type", "")
                        if type == "error":
                            st.error(f"Agent Error: {data['error']}")
                            break

                        elif type == "progress":
                            # Update progress bar
                            progress_bar.progress(data["value"])
                            progress_text.text(data["text"])
                        
                        else:
                            final_verdict = data['final_verdict']
            
            # 6. After the loop, save the final state and rerun
            if final_verdict:
                claim_result = final_verdict.get("claim_verdict", {})

                st.success("Claim verified!")
                st.session_state.workflow_complete = True
                st.session_state.claim_verdict = claim_result
                st.session_state.verifying_claim = False

                if isinstance(claim_result, dict):
                    claim_text = st.session_state.claim
                    verdict_flag = claim_result.get("pass_to_materials_agent", False)

                    # Ensure we always operate on a list copy to avoid Streamlit mutation warnings
                    current_claims = list(st.session_state.materials_verified_claims)

                    def _confidence_from_verdict(verdict: str) -> float:
                        verdict_upper = (verdict or "").upper()
                        if verdict_upper == "TRUE":
                            return 0.9
                        if verdict_upper == "FALSE":
                            return 0.1
                        return 0.5

                    if verdict_flag:
                        verified_entry = {
                            "claim_id": claim_result.get("claim_id") or f"claim_{len(current_claims) + 1:03d}",
                            "claim": claim_text,
                            "verdict": claim_result.get("overall_verdict", "UNKNOWN"),
                            "confidence": claim_result.get("confidence", _confidence_from_verdict(claim_result.get("overall_verdict"))),
                            "explanation": claim_result.get("explanation", ""),
                            "evidence": claim_result.get("main_evidence", []),
                            "pass_to_materials_agent": True,
                        }

                        # Update existing entry for the same claim text if present
                        updated = False
                        for idx, existing in enumerate(current_claims):
                            if existing.get("claim") == claim_text:
                                current_claims[idx] = verified_entry
                                updated = True
                                break
                        if not updated:
                            current_claims.append(verified_entry)

                        st.session_state.materials_verified_claims = current_claims
                        st.session_state.verified_claims = current_claims
                        st.info("Claim flagged for materials generation.")
                    else:
                        filtered = [c for c in current_claims if c.get("claim") != claim_text]
                        if len(filtered) != len(current_claims):
                            st.session_state.materials_verified_claims = filtered
                            st.session_state.verified_claims = filtered
                            st.warning("Claim removed from materials queue (not approved by fact checker).")

                st.rerun()
            else:
                st.error("Failed to get a final verdict from the agent.")

        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to Fact Checker: {str(e)}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
    # ===== END OF MODIFIED PART =====


# --- Display Verdicts ---
if st.session_state.workflow_complete and st.session_state.claim_verdict:
    st.header("Claim Verdict")
    clm = st.session_state.claim_verdict
    original_claim = st.session_state.claim

    st.markdown(f"""
    <div class="material-card">
        <h4>{original_claim}</h4>
        <p><strong>Overall Verdict:</strong> {clm.get('overall_verdict', 'N/A')}</p>
        <p><strong>Reasoning:</strong> {clm.get('explanation', 'N/A')}</p>
        <p><strong>Evidence Used:</strong> {', '.join([ev['source'] for ev in clm.get('main_evidence', [])])}</p>
        <p><strong>Should you pass to Materials Agent:</strong> {"Yes" if clm.get('pass_to_materials_agent') else "No"}
        </p>
    </div>
    """, unsafe_allow_html=True)

# --- Continue Button ---
if st.session_state.workflow_complete:
    st.header("Continue to Materials Generation")
    st.write("If you would like to pass this to the materials agent, click the button below to proceed.")
    if st.button(
        "Generate Materials from Claim", 
        type="primary",
        disabled=not st.session_state.workflow_complete
    ):
        # --- Pass to materials agent ---
        with st.spinner("Passing selected claims to Materials Agent..."):
            st.switch_page("pages/3_Marketing_Decision.py")

# Footer
st.markdown("---")
st.markdown("**Fact Checker Agent** | Part of the Rags2Riches AI Sales Assistant Suite")

# Debug information (only show in development)
if st.checkbox("Show Debug Info"):
    st.subheader("Debug Information")
    st.write("Session State:")
    st.json(dict(st.session_state))