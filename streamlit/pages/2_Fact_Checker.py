import streamlit as st
import requests  
import json      
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo 

# Streamlit page config
st.set_page_config(
    page_title="Fact Checker",
    # page_icon="🎯",
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

# --- Session State Initialization (Good, keep this) ---
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'claim' not in st.session_state:
    st.session_state.claim = "Shopee has a terrible working culture"
if 'workflow_complete' not in st.session_state:
    st.session_state.workflow_complete = False
if 'claim_verdict' not in st.session_state:
    st.session_state.claim_verdict = ""

# --- UI (Good, keep all this) ---
st.markdown(f"""
<div class="main-header">
    <h1>Fact Checking Agent</h1>
    <p>Fact checking the claims from the RAG for your convenience</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("📋 Session Setup")
    salesperson_id = st.text_input("Salesperson ID", value="SP12345")
    client_context = st.text_area(
        "Client Context",
        value="Small e-commerce startup in Singapore...",
        height=200
    )
    if st.button("Reset Session"):
        # ... (reset logic is fine, keep it) ...
        st.rerun()

# --- Main Content Area ---
if 'claim' in st.session_state:
    st.header("Claim To Verify")
    st.write(f"Claim: {st.session_state.claim}")

    # ===== THIS IS THE MODIFIED PART =====
    if st.button("Verify claim", type="primary"):
        
        # --- This is the Sample Request you asked for ---
        # 1. Define the API endpoint
        API_URL = "http://fastapi_service:8000/check-claim"
        
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

        try:
            # 4. Call the API with stream=True
            with requests.post(API_URL, json=payload, stream=True, timeout=300) as response:
                response.raise_for_status() # Raise an error for bad responses
                
                # 5. Iterate over the streaming response line-by-line
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line.decode('utf-8'))
                        
                        # Check for an error message from the agent
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
                st.success("Claim verified!")
                st.session_state.workflow_complete = True
                st.session_state.claim_verdict = final_verdict
                st.rerun()
            else:
                st.error("Failed to get a final verdict from the agent.")

        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to Fact Checker: {str(e)}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
    # ===== END OF MODIFIED PART =====


# --- Display Verdicts (No changes needed, this is perfect) ---
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

# --- Continue Button (No changes needed, this is perfect) ---
if st.session_state.workflow_complete:
    st.header("Continue to Materials Generation")
    st.write("If you would like to pass this to the materials agent, click the button below to proceed.")
    if st.button(
        "Generate Materials from Claim", 
        type="primary",
        disabled=not st.session_state.workflow_complete
    ):
        # --- Placeholder for passing to materials agent ---
        with st.spinner("Passing selected claims to Materials Agent..."):
            # In a real scenario, you would call the pass_to_materials function here
            # with the selected claims.
            # For now, we'll just show an info message.
            st.info(f"Placeholder: Passing claim to the Materials Generation Agent.")
            # ----------------------------------------------------

# Footer
st.markdown("---")
st.markdown("**Fact Checker Agent** | Part of the Rags2Riches AI Sales Assistant Suite")

# Debug information (only show in development)
if st.checkbox("Show Debug Info"):
    st.subheader("Debug Information")
    st.write("Session State:")
    st.json(dict(st.session_state))