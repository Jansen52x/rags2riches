"""
Materials Decision UI - Streamlit-based interface for salespeople
to review and approve presentation materials recommendations
"""

import streamlit as st
import json
import uuid
from datetime import datetime
import requests

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

# Header
st.markdown(f"""
<div class="main-header">
    <h1>Meeting Materials Decision Agent</h1>
    <p>Presentation materials recommendation and management</p>
</div>
""", unsafe_allow_html=True)

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
if 'verified_claims' in st.session_state:
    st.header("Current Claims")
    for i, claim in enumerate(st.session_state.verified_claims):
        with st.expander(f"Claim {i+1}: {claim['claim_text'][:60]}..."):
            st.write(f"**Confidence:** {claim['confidence']:.0%}")
    
    # Generate materials button
    if st.button("🎨 Generate Marketing Materials", type="primary"):
        with st.spinner("Analyzing claims and generating materials..."):
            try:
                st.info("Attempting to connect to FastAPI service...")
                # Call FastAPI endpoint
                payload = {
                    "verified_claims": st.session_state.verified_claims,
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
                    "http://fastapi_service:8001/generate-materials",
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
                    
                    # Show any generated files
                    if result.get("generated_files"):
                        st.header("🖼️ Generated Materials")
                        cols = st.columns(2)
                        for i, file_path in enumerate(result["generated_files"]):
                            col = cols[i % 2]
                            with col:
                                if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                                    st.image(f"http://fastapi_service:8001{file_path}")
                                elif file_path.lower().endswith('.mp4'):
                                    st.video(f"http://fastapi_service:8001{file_path}")
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
        response = requests.get("http://fastapi_service:8001/generated-files")
        if response.ok:
            files = response.json().get("files", [])
            if files:
                st.header("🖼️ Current Generated Materials")
                cols = st.columns(2)
                for i, file in enumerate(files):
                    col = cols[i % 2]
                    with col:
                        name = file["name"]
                        url = file["url"]
                        if name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            st.image(f"http://fastapi_service:8001{url}", caption=name)
                        elif name.lower().endswith('.mp4'):
                            st.video(f"http://fastapi_service:8001{url}")
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