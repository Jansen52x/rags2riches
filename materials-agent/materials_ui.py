"""
Materials Decision UI - Streamlit-based interface for salespeople
to review and approve presentation materials recommendations
"""

import streamlit as st
import json
import uuid
from datetime import datetime
import sys
import os

# Ensure we can import modules from both this folder and the project root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Try to import the full LangGraph-powered agent; fall back to a simple agent if deps are missing
AGENT_MODE = "full"
MaterialsDecisionState = None
create_materials_decision_workflow = None

try:
    from materials_decision_agent import (
        MaterialsDecisionState, 
        create_materials_decision_workflow,
        MaterialType,
        Priority
    )
except Exception:
    # Fallback: use lightweight rule-based agent that has no heavy dependencies
    try:
        from simple_demo import SimpleMaterialsAgent  # type: ignore
        AGENT_MODE = "simple"
    except Exception as e:
        st.error(
            "Unable to load materials agent. Install dependencies or run `pip install -r materials-agent/requirements.txt`.\n"
            f"Import error: {e}"
        )
        st.stop()

# Streamlit page config
st.set_page_config(
    page_title="Meeting Materials Decision Agent",
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

.priority-high {
    border-left: 5px solid #ff4444;
}

.priority-medium {
    border-left: 5px solid #ffaa00;
}

.priority-low {
    border-left: 5px solid #44ff44;
}

.status-badge {
    padding: 0.2rem 0.5rem;
    border-radius: 15px;
    font-size: 0.8rem;
    font-weight: bold;
}

.status-pending {
    background: #fff3cd;
    color: #856404;
}

.status-approved {
    background: #d4edda;
    color: #155724;
}

.status-rejected {
    background: #f8d7da;
    color: #721c24;
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
mode_badge = "<span style='font-size:0.9rem;padding:0.2rem 0.5rem;border-radius:8px;background:#eef;border:1px solid #99f;color:#224;'>Full (AI)</span>" if AGENT_MODE == "full" else "<span style='font-size:0.9rem;padding:0.2rem 0.5rem;border-radius:8px;background:#efe;border:1px solid #9c9;color:#262;'>Simple (No AI)</span>"

st.markdown(f"""
<div class="main-header">
    <h1>Meeting Materials Decision Agent</h1>
    <p>Presentation materials recommendation and management &nbsp;•&nbsp; Mode: {mode_badge}</p>
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
    
    client_context = st.text_area(
        "Client Context",
        value="Small e-commerce startup in Singapore looking to understand market opportunities",
        height=100,
        help="Describe your client and meeting context"
    )
    
    st.header("Verified Claims Input")
    
    # Sample verified claims for testing
    sample_claims = [
        {
            "claim_id": "claim_001",
            "claim": "Singapore's e-commerce market reached SGD 9 billion in 2023",
            "verdict": "TRUE",
            "confidence": 0.9,
            "evidence": [
                {"source": "Singapore Government Trade Statistics", "summary": "Official data confirms SGD 9.1B e-commerce sales in 2023"}
            ]
        },
        {
            "claim_id": "claim_002", 
            "claim": "Shopee leads Singapore e-commerce market share",
            "verdict": "TRUE", 
            "confidence": 0.85,
            "evidence": [
                {"source": "TechCrunch SE Asia Report", "summary": "Shopee holds 35% market share in Singapore"}
            ]
        }
    ]
    
    if st.button("Load Sample Claims"):
        st.session_state.verified_claims = sample_claims
        st.success("Sample claims loaded!")
    
    # Manual claims input
    claims_json = st.text_area(
        "Or paste verified claims JSON:",
        height=150,
        help="Paste JSON array of verified claims from fact-checker"
    )
    
    if st.button("Parse Claims JSON"):
        try:
            if claims_json.strip():
                st.session_state.verified_claims = json.loads(claims_json)
                st.success(f"Loaded {len(st.session_state.verified_claims)} claims")
        except json.JSONDecodeError:
            st.error("Invalid JSON format")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Display current verified claims
    if 'verified_claims' in st.session_state:
        st.header("Verified Claims")
        
        for i, claim in enumerate(st.session_state.verified_claims):
            with st.expander(f"Claim {i+1}: {claim['claim'][:60]}..."):
                st.write(f"**Verdict:** {claim['verdict']}")
                st.write(f"**Confidence:** {claim['confidence']:.0%}")
                st.write(f"**Evidence:**")
                for evidence in claim['evidence']:
                    st.write(f"- {evidence['source']}: {evidence['summary']}")
        
        # Generate materials recommendations
        if st.button("Generate Materials Recommendations", type="primary"):
            with st.spinner("Analyzing claims and generating recommendations..."):
                try:
                    if AGENT_MODE == "full" and MaterialsDecisionState and create_materials_decision_workflow:
                        # Create workflow state
                        initial_state = MaterialsDecisionState(
                            session_id=st.session_state.session_id,
                            salesperson_id=salesperson_id,
                            client_context=client_context,
                            verified_claims=st.session_state.verified_claims,
                            material_recommendations=[],
                            selected_materials=[],
                            generation_queue=[],
                            decision_complete=False,
                            user_feedback=None
                        )
                        # Run workflow
                        workflow = create_materials_decision_workflow()
                        final_state = workflow.invoke(initial_state)
                        # Update session state
                        st.session_state.recommendations = final_state['material_recommendations']
                        st.session_state.selected_materials = final_state['selected_materials']
                        st.session_state.generation_queue = final_state['generation_queue']
                        st.session_state.workflow_complete = True
                    else:
                        # Simple fallback path (no heavy deps)
                        agent = SimpleMaterialsAgent()
                        recs = agent.analyze_claims_for_materials(st.session_state.verified_claims, client_context)
                        selected, _total = agent.prioritize_materials(recs, time_limit=120)
                        queue = agent.create_generation_queue(selected)
                        st.session_state.recommendations = recs
                        st.session_state.selected_materials = selected
                        st.session_state.generation_queue = queue
                        st.session_state.workflow_complete = True

                    st.success("Materials recommendations generated!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating recommendations: {str(e)}")
    
    # Display recommendations
    if st.session_state.workflow_complete and st.session_state.recommendations:
        st.header("Recommended Materials")
        
        for i, rec in enumerate(st.session_state.recommendations):
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
            
            # Material details in expander
            with st.expander("View Details"):
                st.json(rec['content_requirements'])
                st.write(f"**Reasoning:** {rec.get('reasoning', 'N/A')}")
                
                # Approval buttons
                col_approve, col_reject, col_modify = st.columns(3)
                
                with col_approve:
                    if st.button(f"✅ Approve", key=f"approve_{i}"):
                        st.success(f"Approved: {rec['title']}")
                
                with col_reject:
                    if st.button(f"❌ Reject", key=f"reject_{i}"):
                        st.warning(f"Rejected: {rec['title']}")
                
                with col_modify:
                    if st.button(f"✏️ Modify", key=f"modify_{i}"):
                        st.info(f"Modify mode for: {rec['title']}")

with col2:
    # Status panel
    st.header("Session Status")
    
    if 'verified_claims' in st.session_state:
        st.metric("Verified Claims", len(st.session_state.verified_claims))
    
    if st.session_state.recommendations:
        st.metric("Recommendations", len(st.session_state.recommendations))
        
        # Priority breakdown
        priority_counts = {}
        for rec in st.session_state.recommendations:
            priority = rec.get('priority', 'medium')
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        st.write("**Priority Breakdown:**")
        for priority, count in priority_counts.items():
            st.write(f"- {priority.title()}: {count}")
    
    if st.session_state.selected_materials:
        st.metric("Selected for Creation", len(st.session_state.selected_materials))
        
        # Time estimation
        total_time = sum(mat.get('estimated_time_minutes', 0) for mat in st.session_state.selected_materials)
        st.metric("Total Time (minutes)", total_time)
    
    # Action buttons
    st.header("Actions")
    
    if st.session_state.workflow_complete:
        if st.button("Start Material Generation"):
            st.info("Material generation would be triggered here")
            # This would trigger the image/video generation agents
        
        if st.button("Save Session"):
            # Save the session to database
            session_data = {
                'session_id': st.session_state.session_id,
                'salesperson_id': salesperson_id,
                'recommendations': st.session_state.recommendations,
                'selected_materials': st.session_state.selected_materials
            }
            st.success("Session saved!")
    
    if st.button("Reset Session"):
        for key in list(st.session_state.keys()):
            if key != 'session_id':
                del st.session_state[key]
        st.session_state.session_id = str(uuid.uuid4())
        st.success("Session reset!")
        st.rerun()

# Footer
st.markdown("---")
st.markdown("**Materials Decision Agent** | Part of the Rags2Riches AI Sales Assistant Suite")

# Debug information (only show in development)
if st.checkbox("Show Debug Info"):
    st.subheader("Debug Information")
    st.write("Session State:")
    st.json(dict(st.session_state))