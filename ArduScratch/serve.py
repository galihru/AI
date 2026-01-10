#!/usr/bin/env python3
"""Simple web UI for Arduino code generator."""
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import os
import sys

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from generate_project import load_model, generate, retrieve_context
from tokenizers import Tokenizer
import torch

app = FastAPI(title="ArduScratch AI")

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), 'static')
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Global state
MODEL = None
TOKENIZER = None
INDEX_FILE = "data/index.json"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class GenerateRequest(BaseModel):
    project: str
    spec: str
    temperature: float = 0.8
    max_tokens: int = 1024


class GenerateResponse(BaseModel):
    project: str
    code: str
    status: str


@app.on_event("startup")
async def startup():
    global MODEL, TOKENIZER
    print("Loading AI model...")
    MODEL, _ = load_model("models/latest", DEVICE)
    TOKENIZER = Tokenizer.from_file("data/tokenizer/tokenizer.json")
    print(f"✓ AI ready on {DEVICE}")


@app.get("/")
async def root():
    # Serve HTML page if exists
    html_file = os.path.join(os.path.dirname(__file__), 'static', 'index.html')
    if os.path.exists(html_file):
        return FileResponse(html_file)
    
    # Fallback to JSON
    return {
        "name": "ArduScratch AI",
        "status": "running",
        "device": DEVICE,
        "endpoints": ["/generate", "/health"]
    }


@app.get("/api")
async def api_info():
    return {
        "name": "ArduScratch AI",
        "status": "running",
        "device": DEVICE,
        "endpoints": ["/generate", "/health"]
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.post("/generate", response_model=GenerateResponse)
async def generate_code(req: GenerateRequest):
    if MODEL is None or TOKENIZER is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Retrieve context
        context = retrieve_context(INDEX_FILE, req.spec)
        
        # Create prompt
        prompt = f"""// Arduino Project: {req.project}
// Specification: {req.spec}

// Reference code:
{context[:300]}

// Generated implementation:
"""
        
        # Generate
        code = generate(
            MODEL, 
            TOKENIZER, 
            prompt, 
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            device=DEVICE
        )
        
        return GenerateResponse(
            project=req.project,
            code=code,
            status="success"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("=" * 50)
    print("ArduScratch AI Server")
    print("=" * 50)
    print("Starting server at http://127.0.0.1:8000")
    print("API docs: http://127.0.0.1:8000/docs")
    print("=" * 50)
    
    uvicorn.run(app, host="127.0.0.1", port=8000)
