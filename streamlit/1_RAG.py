import streamlit as st
import uuid

# Initialize claim at the very top
if 'claim' not in st.session_state:
    st.session_state.claim = ""
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    

claim_input = st.text_input(
    "Enter your claim here:", 
    value=st.session_state.claim,
    key="claim_input"  # Different key to avoid conflicts
)

st.session_state.claim = claim_input

if st.button("Fact-Check Claim"):
    st.write(f"Fact-checking the claim: {st.session_state.claim}")
    st.switch_page("pages/2_Fact_Checker.py")