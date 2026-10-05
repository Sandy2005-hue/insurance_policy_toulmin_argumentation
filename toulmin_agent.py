import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from shared_retriever import get_hybrid_retriever

load_dotenv()

hybrid_search = get_hybrid_retriever(k=4)
llm = ChatGoogleGenerativeAI(model="gemini-3.7-flash", temperature=0.1)


def quote_in_context(quote, docs):
    if not quote or quote == "No relevant exclusions found.":
        return True
    norm = lambda s: " ".join(s.lower().split())
    normalized_quote = norm(quote)
    return any(normalized_quote in norm(d.page_content) for d in docs)

def run_toulmin_pipeline(patient_claim: str):
    
    coverage_docs = hybrid_search(f"benefits coverage allowed procedures: {patient_claim}")
    exclusion_docs = hybrid_search(f"waiting periods exclusions non-payable: {patient_claim}")
    
    
    coverage_context = "\n\n".join([f"[Citation: {d.metadata.get('citation', 'Unknown')}]\n{d.page_content}" for d in coverage_docs])
    exclusion_context = "\n\n".join([f"[Citation: {d.metadata.get('citation', 'Unknown')}]\n{d.page_content}" for d in exclusion_docs])

   
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Insurance Underwriting AI. 
        Analyze the patient's claim against the provided policy context and output a 5-part Toulmin argument in JSON.
        
        RULES:
        1. Base your decision ONLY on the provided context. If information is missing, output "Needs Human Review".
        2. 'Backing' MUST quote the exact text from the Coverage Context.
        3. 'Rebuttal' MUST quote the exact text from the Exclusion Context. If none apply, state "No relevant exclusions found."
        4. 'Claim' MUST be either "Claim Approved", "Claim Denied", or "Needs Human Review".
        
        OUTPUT FORMAT (JSON ONLY):
        {{
            "Grounds": "Summary of the patient's claim",
            "Warrant": "The logical policy rule applied",
            "Backing": "Exact quote from coverage context",
            "Rebuttal": "Exact quote from exclusions/waiting periods",
            "Claim": "Final decision (Claim Approved / Claim Denied / Needs Human Review)"
        }}"""),
        ("user", """
        COVERAGE CONTEXT:\n{coverage}\n
        EXCLUSION CONTEXT:\n{exclusion}\n
        PATIENT CLAIM: {claim}
        """)
    ])

    chain = prompt | llm | JsonOutputParser()
    result = chain.invoke({
        "coverage": coverage_context,
        "exclusion": exclusion_context,
        "claim": patient_claim
    })
    
    
    all_docs = coverage_docs + exclusion_docs
    backing_verified = quote_in_context(result.get("Backing", ""), all_docs)
    rebuttal_verified = quote_in_context(result.get("Rebuttal", ""), all_docs)
    
    result["Guardrail_Status"] = "Verified" if (backing_verified and rebuttal_verified) else "Unverified (Hallucination Detected)"
    
    return result