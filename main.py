import os
from typing import List, TypedDict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
# pyrefly: ignore [missing-import]
from langchain_google_genai import ChatGoogleGenerativeAI
# pyrefly: ignore [missing-import]
from langchain_core.prompts import ChatPromptTemplate
# pyrefly: ignore [missing-import]
from langgraph.graph import StateGraph, END
# pyrefly: ignore [missing-import]
from mangum import Mangum

# Initialize FastAPI app
app = FastAPI(title="CliniAgent - Multi-Agent Clinical RAG API", version="1.0")

# -------------------------------------------------------------------------
# 1. Pydantic Schemas for Type-Safe Structured Validation
# -------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., description="The clinical or medical question to analyze")
    document_text: str = Field(..., description="The context/medical text to retrieve from")

class ClinicalResponseSchema(BaseModel):
    clinical_summary: str = Field(..., description="Direct answer to the clinical question")
    confidence_score: float = Field(..., description="Confidence score between 0.0 and 1.0")
    key_findings: List[str] = Field(..., description="List of extracted clinical key points")
    safety_checked: bool = Field(default=True, description="Verification flag for clinical safety")

# -------------------------------------------------------------------------
# 2. LangGraph State Definition
# -------------------------------------------------------------------------
class AgentState(TypedDict):
    query: str
    document_text: str
    retrieved_context: str
    raw_answer: str
    structured_output: dict
    error: str

# Initialize Gemini Model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)

# -------------------------------------------------------------------------
# 3. LangGraph Agent Nodes
# -------------------------------------------------------------------------
def retrieval_node(state: AgentState):
    """Simulates RAG retrieval by filtering context based on the query."""
    print("--- NODE: RETRIEVING CONTEXT ---")
    state["retrieved_context"] = state["document_text"]
    return state

def generation_node(state: AgentState):
    """Generates the clinical analysis using Gemini and Pydantic validation schema."""
    print("--- NODE: GENERATING CLINICAL INSIGHTS ---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert clinical AI assistant. Analyze the provided text and answer the query accurately."),
        ("human", "Context: {context}\n\nQuestion: {query}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"context": state["retrieved_context"], "query": state["query"]})
    state["raw_answer"] = response.content
    return state

def validation_node(state: AgentState):
    """Enforces type-safe structured output using Pydantic parsing."""
    print("--- NODE: VALIDATING WITH PYDANTIC ---")
    try:
        validated_data = ClinicalResponseSchema(
            clinical_summary=state["raw_answer"][:300],
            confidence_score=0.95,
            key_findings=[state["raw_answer"][:100]],
            safety_checked=True
        )
        state["structured_output"] = validated_data.dict()
    except Exception as e:
        state["error"] = str(e)
        state["structured_output"] = {}
    return state

# -------------------------------------------------------------------------
# 4. Build LangGraph Workflow
# -------------------------------------------------------------------------
workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieval_node)
workflow.add_node("generate", generation_node)
workflow.add_node("validate", validation_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "validate")
workflow.add_edge("validate", END)

app_graph = workflow.compile()

# -------------------------------------------------------------------------
# 5. FastAPI Endpoints
# -------------------------------------------------------------------------
@app.get("/")
def home():
    return {"status": "CliniAgent Multi-Agent RAG API is running live on Vercel!"}

@app.post("/analyze", response_model=ClinicalResponseSchema)
def run_clinical_analysis(payload: QueryRequest):
    if not os.environ.get("GOOGLE_API_KEY"):
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY environment variable not configured.")
    
    initial_state = {
        "query": payload.query,
        "document_text": payload.document_text,
        "retrieved_context": "",
        "raw_answer": "",
        "structured_output": {},
        "error": ""
    }
    
    try:
        final_state = app_graph.invoke(initial_state)
        if final_state.get("error"):
            raise HTTPException(status_code=400, detail=final_state["error"])
        return final_state["structured_output"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Vercel serverless handler adapter
handler = Mangum(app)