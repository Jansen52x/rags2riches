import streamlit as st
import uuid
import requests
import json
from datetime import datetime

# TO ADD: 
# 1) Fact-Check this Claim button
# 2) Edit claim before sending text input
# - input = st.text_input("Claim to fact-check", value=st.session_state.claim, key="edit_claim_input")
# - st.session_state.claim = input
# 3) Send to Fact Checker button (switch_page)
# - switch_page("pages/2_Fact_Checker.py")


# Streamlit page config
st.set_page_config(
    page_title="RAG Chat Service",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.header("🔍 Query Knowledge Base")
st.markdown("Search through ingested company data, documents, and materials.")
st.markdown("")  # Add spacing

# RAG service configuration
RAG_SERVICE_URL = "http://fastapi_service:8001/query_rag"

# Query input section with better spacing
col_query, col_spacer, col_settings = st.columns([2.5, 0.2, 1.5])

with col_query:
    st.markdown("### Enter Your Question")
    query_text = st.text_area(
        "Query",
        placeholder="e.g., Tell me about companies in the robotics industry",
        height=120,
        label_visibility="collapsed"
    )
    
    st.markdown("")  # Spacing
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    with col_btn2:
        clear_button = st.button("🗑️ Clear Results", use_container_width=True)

with col_spacer:
    st.empty()

with col_settings:
    st.markdown("### Search Settings")
    st.markdown("")  # Spacing

    num_results = st.slider("Number of results", min_value=1, max_value=10, value=5)
    
    st.markdown("")  # Spacing

    include_sources = st.checkbox("Show source documents", value=True)

    st.markdown("")  # Spacing
    st.markdown("### Filters (Optional)")
    st.markdown("")  # Spacing
    
    filter_industry = st.text_input("Industry", placeholder="e.g., robotics")
    filter_company = st.text_input("Company name", placeholder="e.g., Williams-Frederick")
    filter_source = st.selectbox("Source type", ["All", "synthetic_data", "pdf", "csv", "image"])

# Clear results
if clear_button:
    if 'rag_results' in st.session_state:
        del st.session_state.rag_results
    st.rerun()

# Perform search
if search_button and query_text.strip():
    with st.spinner("Searching knowledge base..."):
        try:
            # Build filters
            filters = {}
            if filter_industry:
                filters["industry"] = filter_industry
            if filter_company:
                filters["company_name"] = filter_company
            if filter_source != "All":
                filters["source"] = filter_source

            # Determine endpoint
            if filters:
                endpoint = f"{RAG_SERVICE_URL}/builder"
                payload = {
                    "query": query_text,
                    "filters": filters,
                    "k": num_results,
                    "include_sources": include_sources
                }
            else:
                endpoint = f"{RAG_SERVICE_URL}"
                payload = {
                    "query": query_text,
                    "k": num_results,
                    "include_sources": include_sources
                }

            # Make request
            response = requests.post(endpoint, json=payload, timeout=30)
            response.raise_for_status()

            st.session_state.rag_results = response.json()
            st.success("Search completed!")

        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to RAG service. Is it running on port 8001?")
            st.info("Start it with: `cd rag && python main.py`")
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. Try a simpler query.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Display results
if 'rag_results' in st.session_state:
    results = st.session_state.rag_results

    # Display answer
    if 'answer' in results:
        st.markdown("---")
        st.markdown("")  # Spacing
        st.subheader("📝 Answer")
        st.markdown(f"""
        <div style="padding: 2rem; border-radius: 10px; border: 2px solid #000000;  margin: 1rem 0;">
            {results['answer']}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")  # Spacing

        # Only show fact-check option if user wants to extract claims
        with st.expander("⚙️ Verify Claims"):

            # Let user manually select/edit what to fact-check
            claim_input = st.text_input(
                "Enter your claim to verify",
                value="",
                placeholder="e.g., 'Williams-Frederick generated $50M revenue in 2023' or select a specific factual statement from the answer above",
                help="Extract a specific factual claim from the RAG answer that you want to verify with external sources"
            )

            col_btn1, col_btn2 = st.columns([1, 1])
            with col_btn1:
                if st.button("🔍 Fact-Check This Claim", type="primary", use_container_width=True, disabled=not claim_input.strip()):
                    st.session_state.claim = claim_input.strip()
                    st.session_state.rag_context = f"Original Query: {query_text}\n\nRAG Answer Context: {results['answer'][:200]}..."
                    st.switch_page("pages/2_Fact_Checker.py")
            with col_btn2:
                if st.button("📋 Use Full Answer", use_container_width=True):
                    st.session_state.claim = results['answer']
                    st.session_state.rag_context = query_text
                    st.switch_page("pages/2_Fact_Checker.py")

        st.markdown("")  # Spacing

    # Display sources
    if include_sources and 'sources' in results and results['sources']:
        st.markdown("---")
        st.markdown("")  # Spacing
        st.subheader(f"📚 Source Documents ({len(results['sources'])})")
        st.markdown("")  # Spacing

        for i, source in enumerate(results['sources'], 1):
            metadata = source.get('metadata', {})
            score = source.get('score', 0)
            rerank_score = source.get('rerank_score')

            # Create expandable card for each source
            with st.expander(f"Source {i}: {metadata.get('company_name', 'Document')} (Score: {score:.3f})"):
                st.markdown("")  # Spacing inside expander
                
                # Metadata
                col_meta1, col_meta2 = st.columns(2)
                with col_meta1:
                    if 'company_name' in metadata:
                        st.write(f"**Company:** {metadata['company_name']}")
                    if 'industry' in metadata:
                        st.write(f"**Industry:** {metadata['industry']}")
                    if 'source' in metadata:
                        st.write(f"**Source:** {metadata['source']}")

                with col_meta2:
                    st.write(f"**Similarity Score:** {score:.3f}")
                    if rerank_score is not None:
                        st.write(f"**Rerank Score:** {rerank_score:.3f}")
                    if 'contact_email' in metadata:
                        st.write(f"**Contact:** {metadata['contact_email']}")

                st.markdown("")  # Spacing
                
                # Content
                st.markdown("**Content:**")
                st.text_area(
                    "Document content",
                    value=source.get('content', 'No content available'),
                    height=150,
                    key=f"source_{i}",
                    label_visibility="collapsed"
                )

                st.markdown("")  # Spacing

                # Action buttons
                col_action1, col_action2 = st.columns(2)
                with col_action1:
                    if st.button(f"📋 Copy to Materials", key=f"copy_{i}"):
                        st.info(f"Added source {i} to materials context")
                with col_action2:
                    # Option to extract claim from this source
                    if st.button(f"🔍 Extract Claim", key=f"extract_{i}"):
                        st.session_state[f'extract_from_source_{i}'] = True
                        st.rerun()

                # Show claim extraction UI if user clicked extract
                if st.session_state.get(f'extract_from_source_{i}', False):
                    st.markdown("---")
                    st.markdown("**Extract a verifiable claim from this source:**")
                    extracted_claim = st.text_area(
                        "Claim to verify",
                        value="",
                        placeholder="e.g., 'Company X achieved 25% growth in Q4 2023'",
                        key=f"claim_extract_{i}",
                        height=80
                    )
                    col_e1, col_e2 = st.columns(2)
                    with col_e1:
                        if st.button("✅ Verify This", key=f"verify_{i}"):
                            if extracted_claim.strip():
                                st.session_state.claim = extracted_claim.strip()
                                st.session_state.rag_context = f"Query: {query_text}\nSource: {metadata.get('company_name', 'Unknown')}\nDocument Score: {score:.3f}"
                                st.switch_page("pages/2_Fact_Checker.py")
                    with col_e2:
                        if st.button("❌ Cancel", key=f"cancel_{i}"):
                            st.session_state[f'extract_from_source_{i}'] = False
                            st.rerun()
            
            st.markdown("")  # Spacing between expanders

        # Export options
        st.markdown("---")
        st.markdown("")  # Spacing
        col_export1, col_export2 = st.columns(2)
        with col_export1:
            if st.button("💾 Save Results to Session"):
                st.success("Results saved to current session!")
        with col_export2:
            # Export as JSON
            json_str = json.dumps(results, indent=2)
            st.download_button(
                label="⬇️ Download Results (JSON)",
                data=json_str,
                file_name=f"rag_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

    # Quick actions
    st.markdown("---")
    st.markdown("")  # Spacing
    st.subheader("⚡ Quick Actions")
    st.markdown("")  # Spacing

    col_quick1, col_quick2, col_quick3 = st.columns(3)

    with col_quick1:
        if st.button("🔄 Refine Query"):
            st.info("Edit your query above and search again")

    with col_quick2:
        if st.button("📊 View Stats"):
            try:
                stats_response = requests.get(f"{RAG_SERVICE_URL}/stats", timeout=5)
                if stats_response.ok:
                    stats = stats_response.json()
                    st.json(stats)
            except:
                st.error("Could not fetch stats")

    with col_quick3:
        if st.button("📖 View Templates"):
            try:
                templates_response = requests.get(f"{RAG_SERVICE_URL}/query/templates", timeout=5)
                if templates_response.ok:
                    templates = templates_response.json()
                    st.json(templates)
            except:
                st.error("Could not fetch templates")