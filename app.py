import streamlit as st
from toulmin_agent import run_toulmin_pipeline


st.set_page_config(page_title="Enterprise AI: Insurance Claim Adjudicator", layout="wide")

st.title("🛡️ Health Insurance Claim Adjudicator")
st.subheader("Generative AI Agentic RAG Pipeline (Toulmin Logic)")

st.markdown("---")

# User Input
claim_input = st.text_area(
    "Enter the Patient's Claim (Grounds):", 
    placeholder="e.g., The patient underwent Cataract surgery after holding the policy for 12 months. Total bill: Rs 45,000.",
    height=100
)

if st.button("Process Claim"):
    if claim_input:
        with st.spinner("Executing Agentic Retrieval & Toulmin Synthesis..."):
            try:
            
                response = run_toulmin_pipeline(claim_input)
                
                
                st.success("Analysis Complete")
                

                decision_color = "green" if "Approved" in response.get("Claim", "") else "red"
                st.markdown(f"### Final Verdict: <span style='color:{decision_color}'>{response.get('Claim', 'N/A')}</span>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info(f"**1. Grounds (Data):**\n\n{response.get('Grounds', '')}")
                    st.warning(f"**3. Rebuttal (Exclusions/Limits):**\n\n{response.get('Rebuttal', '')}")
                
                with col2:
                    st.success(f"**2. Warrant (Logical Rule):**\n\n{response.get('Warrant', '')}")
                    st.write(f"**4. Backing (Policy Citation):**\n\n{response.get('Backing', '')}")
                    
            except Exception as e:
                st.error(f"Pipeline Execution Failed: {str(e)}")
    else:
        st.warning("Please enter a claim to process.")