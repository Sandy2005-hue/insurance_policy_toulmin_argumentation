import streamlit as st
from toulmin_agent import run_toulmin_pipeline

st.set_page_config(page_title="Insurance Adjudicator", layout="wide")

st.title("🛡️ Health Insurance Claim Adjudicator")
st.subheader("Hybrid Agentic RAG with Hallucination Guardrails")

st.markdown("---")

claim_input = st.text_area(
    "Enter the Patient's Claim (Grounds):", 
    placeholder="e.g., The patient underwent Cataract surgery after holding the policy for 12 months. Total bill: Rs 45,000.",
    height=100
)

if st.button("Process Claim"):
    if claim_input:
        with st.spinner("Executing Hybrid Retrieval & Guardrail Checks..."):
            try:
                response = run_toulmin_pipeline(claim_input)
                
                # Verdict Colors
                verdict = response.get('Claim', 'N/A')
                if "Approved" in verdict: color = "green"
                elif "Denied" in verdict: color = "red"
                else: color = "orange"
                
                st.markdown(f"### Final Verdict: <span style='color:{color}'>{verdict}</span>", unsafe_allow_html=True)
                
                # Guardrail Display
                guardrail = response.get("Guardrail_Status", "")
                if "Unverified" in guardrail:
                    st.error(f"🚨 Security Alert: {guardrail}")
                else:
                    st.success(f"✅ Citations: {guardrail}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**1. Grounds:**\n\n{response.get('Grounds', '')}")
                    st.warning(f"**3. Rebuttal:**\n\n{response.get('Rebuttal', '')}")
                with col2:
                    st.success(f"**2. Warrant:**\n\n{response.get('Warrant', '')}")
                    st.write(f"**4. Backing:**\n\n{response.get('Backing', '')}")
                    
            except Exception as e:
                st.error(f"Pipeline Failed: {str(e)}")