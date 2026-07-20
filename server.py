import os
import time
import json
import base64
import pickle
import threading
from typing import Optional, Dict, Any
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama

app = FastAPI(title="KV Global Cache Pre-compiler Worker")

class UpdateRequest(BaseModel):
    client_id: str
    system_prompt: str
    persona: str
    kb: str

# Find the model file
def find_gguf_file() -> Path:
    for path in Path(".").glob("*.gguf"):
        if "mmproj" not in path.name:
            return path
    model_dir = Path("model")
    if model_dir.exists():
        for path in model_dir.glob("*.gguf"):
            if "mmproj" not in path.name:
                return path
    return Path("Qwen3.5-0.8B-Q4_K_M.gguf")

_llm: Optional[Llama] = None
_load_lock = threading.Lock()

def get_llm() -> Llama:
    global _llm
    if _llm is None:
        with _load_lock:
            if _llm is None:
                model_path = find_gguf_file()
                if not model_path.exists():
                    raise FileNotFoundError(f"GGUF model file not found at {model_path}")
                print(f"[KV Worker] Loading model weights from {model_path}...", flush=True)
                _llm = Llama(
                    model_path=str(model_path),
                    n_ctx=8192,
                    n_threads=2,
                    flash_attn=True
                )
    return _llm

def upload_to_redis_placeholder(client_id: str, state_data: bytes) -> bool:
    """
    Placeholder function to mock uploading the binary global state data to Redis.
    In the real implementation, this will perform a redis.set(f"global:{client_id}", state_data) call.
    """
    print(f"[Redis Mock] Saving global state binary for client {client_id} (Size: {len(state_data)} bytes)", flush=True)
    return True

@app.post("/update")
def update_global_cache(req: UpdateRequest) -> Dict[str, Any]:
    try:
        t0 = time.time()
        llm = get_llm()
        
        # Stitch client prompt configuration
        prompt_parts = [
            "System Prompt:",
            req.system_prompt.strip(),
            "",
            "Persona:",
            req.persona.strip(),
            "",
            "Knowledge Base (Authoritative Facts):",
            req.kb.strip(),
            ""
        ]
        stitched_text = "\n".join(prompt_parts)
        
        # Tokenize and evaluate prompt to compile the prefix KV Cache
        tokens = llm.tokenize(stitched_text.encode("utf-8"))
        print(f"[KV Worker] Tokenizing prompt for client {req.client_id} (Token count: {len(tokens)})", flush=True)
        
        llm.reset()
        llm.eval(tokens)
        
        # Save compiled context state
        state_obj = llm.save_state()
        state_bytes = pickle.dumps(state_obj)
        
        # Upload compiled state to Redis (via placeholder)
        upload_to_redis_placeholder(req.client_id, state_bytes)
        
        duration = time.time() - t0
        return {
            "status": "success",
            "client_id": req.client_id,
            "tokens_compiled": len(tokens),
            "compilation_time_seconds": round(duration, 3),
            "state_size_bytes": len(state_bytes)
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Compilation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8002, reload=False)
