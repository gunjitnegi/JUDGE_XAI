from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import shutil
import json
from pathlib import Path
from src.rag.rag_pipeline import RAGPipeline

app = FastAPI(title="JUDGE X AI API", description="Legal Document Summarization & QA API")

pipeline = RAGPipeline()
pipeline.load_index()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Uploads a PDF, processes it, and adds to FAISS index."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    temp_dir = Path("data/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    file_path = temp_dir / file.filename
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        pipeline.ingest_pdf(str(file_path))
        pipeline.save_index()
        return {"status": "success", "message": f"Successfully indexed {file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if file_path.exists():
            os.remove(file_path)

@app.post("/query")
async def query_stream(request: QueryRequest):
    """Executes a RAG query and streams the response chunks as JSON lines."""
    if not pipeline.retriever:
        raise HTTPException(status_code=400, detail="No index loaded. Upload a PDF first.")
        
    def stream_generator():
        try:
            for chunk in pipeline.query(request.query, top_k=request.top_k):
                if isinstance(chunk, dict):
                    # Final output dictionary
                    yield json.dumps({"type": "result", "data": chunk}) + "\n"
                else:
                    # Text chunk
                    yield json.dumps({"type": "text", "data": chunk}) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "data": str(e)}) + "\n"
            
    return StreamingResponse(stream_generator(), media_type="application/x-ndjson")

@app.get("/health")
def health_check():
    return {"status": "ok", "index_loaded": pipeline.retriever is not None}
