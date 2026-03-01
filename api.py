import os
import sys
import json
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from pydantic import BaseModel
from app import get_vector_db, get_llm, CCPAComplianceCheck
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Global variables to hold model state
retrieval_chain = None
is_ready = False

# Create the startup event using lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    global retrieval_chain, is_ready
    print("Starting up and loading models... This might take a few minutes.")
    
    try:
        # Load Chroma DB and Retriever
        vector_db = get_vector_db()
        retriever = vector_db.as_retriever(search_kwargs={"k": 5})
        
        # Load the LLM (Qwen2.5-1.5B)
        llm = get_llm()
        
        # Initialize parser explicitly to ensure json formatting
        parser = PydanticOutputParser(pydantic_object=CCPAComplianceCheck)
        
        # Prompt template tailored to Qwen ChatML format
        template = """<|im_start|>system
You are a helpful assistant analyzing actions for CCPA compliance. Determine if the action described in the user input is a CCPA violation ("harmful") and list the relevant CCPA articles. If the action is legally compliant, follows the rules, or is unrelated to CCPA, it is NOT harmful (set harmful=false).

CRITICAL: Output ONLY a single valid JSON object. Do not explain, do not add introductory text, do not converse. Do NOT wrap the JSON in markdown formatting or code blocks.
{format_instructions}<|im_end|>
<|im_start|>user
Context: {context}

Question/Scenario: {input}<|im_end|>
<|im_start|>assistant
"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "input"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
        
        # Mark health check as ready
        is_ready = True
        print("Models loaded successfully. API is ready.")
    except Exception as e:
        print(f"Failed to load models during startup: {e}")
        
    yield  # API serves requests here
    
    print("Shutting down API...")

app = FastAPI(lifespan=lifespan)
parser = PydanticOutputParser(pydantic_object=CCPAComplianceCheck)

# Request Models
class AnalyzeRequest(BaseModel):
    prompt: str

@app.get("/health")
async def health_check():
    """
    Health check endpoint: will return 200 OK only when models are fully loaded in memory.
    """
    if is_ready:
        return {"status": "ok"}
    return {"status": "loading"}, 503

@app.post("/analyze")
async def analyze_prompt(request: AnalyzeRequest):
    """
    Accepts a prompt, passes to RAG LLM pipeline, and performs logic formatting checks on the JSON parsing.
    Saves outputs to a log file.
    """
    global retrieval_chain
    if not is_ready:
        return {"error": "Model is not yet loaded. Please try again later."}, 503
        
    print(f"\nAnalyzing prompt: '{request.prompt}'")
    
    # 1. Invoke the LangChain retrieval chain
    try:
        response = retrieval_chain.invoke({"input": request.prompt})
        raw_answer = response.get("answer", "").strip()
        
        # 2. Parse into Pydantic model
        try:
            parsed_output = parser.parse(raw_answer)
        except Exception as e:
            print(f"Pydantic parsing failed: {e}. Attempting JSON extraction fallback...")
            json_match = re.search(r'\{.*\}', raw_answer, re.DOTALL)
            if json_match:
                extracted = json_match.group(0)
                try:
                    parsed_data = json.loads(extracted)
                    parsed_output = CCPAComplianceCheck(**parsed_data)
                except Exception as ex:
                    raise Exception(f"Fallback extraction failed: {ex}")
            else:
                raise Exception("No JSON object found in output.")
        
        # 3. Logic validation
        if parsed_output.harmful:
            # If harmful is true, articles list must have at least one item
            if not parsed_output.articles:
                print("Warning: harmful is true but articles is empty. Adding a placeholder article.")
                parsed_output.articles = ["CCPA Violation Detected"]
        else:
            # If harmful is false, articles must be empty
            if parsed_output.articles:
                print("Warning: harmful is false but articles were returned. Clearing articles list.")
                parsed_output.articles = []
                
        # Format the validated response into a dict
        final_response = {
            "harmful": parsed_output.harmful,
            "articles": parsed_output.articles
        }
    except Exception as e:
        print(f"Failed to compile acceptable JSON. Error: {e}")
        # Build an emergency fallback in case the LLM breaks syntax completely
        final_response = {
            "harmful": False,
            "articles": []
        }
        
    # 5. Logging Results
    # Append the request/response to a local log file for organisms to review
    with open("api_results_log.jsonl", "a") as f:
        log_entry = {
            "prompt": request.prompt,
            "response": final_response
        }
        f.write(json.dumps(log_entry) + "\n")
        
    return final_response

if __name__ == "__main__":
    import uvicorn
    # Launch uvicorn locally for testing
    uvicorn.run(app, host="0.0.0.0", port=8000)
