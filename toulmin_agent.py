import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

#no model named 1.5 or 2.5 flash those are taken out by the gemini team itself from AI studios
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.1)

def run_toulmin_pipeline(patient_claim: str):

    coverage_docs = retriever.invoke(f"What are the benefits, coverage, and allowed procedures for: {patient_claim}?")
    exclusion_docs = retriever.invoke(f"What are the waiting periods, exclusions, and non-payable items for: {patient_claim}?")
    
    coverage_context = "\n\n".join([doc.page_content for doc in coverage_docs])
    exclusion_context = "\n\n".join([doc.page_content for doc in exclusion_docs])

   
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Insurance Underwriting AI. 
        Analyze the patient's claim against the provided policy context and output a strict 5-part Toulmin argument in JSON.
        
        RULES:
        1. Base your decision ONLY on the provided context. If the policy does not explicitly cover it, deny it.
        2. 'Backing' MUST quote the exact page or clause from the Coverage Context.
        3. 'Rebuttal' MUST quote the exact page or clause from the Exclusion Context. If none apply, state "No relevant exclusions found."
        4. 'Claim' MUST be either "Claim Approved", "Claim Denied", or "Claim Partially Approved".
        
        OUTPUT FORMAT (JSON ONLY):
        {{
            "Grounds": "Summary of what the patient is asking for",
            "Warrant": "The logical policy rule applied to this scenario",
            "Backing": "Exact quote from coverage context",
            "Rebuttal": "Exact quote from exclusions/waiting periods that might void the claim",
            "Claim": "Final decision (Approved/Denied)"
        }}"""),
        ("user", """
        COVERAGE CONTEXT: {coverage}
        
        EXCLUSION / WAITING PERIOD CONTEXT: {exclusion}
        
        PATIENT CLAIM: {claim}
        """)
    ])

    chain = prompt | llm | JsonOutputParser()
    
    
    result = chain.invoke({
        "coverage": coverage_context,
        "exclusion": exclusion_context,
        "claim": patient_claim
    })
    
    return result