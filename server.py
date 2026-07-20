import os
import time
import json
import base64
import pickle
import threading
import subprocess
import re
import requests
from typing import Optional, Dict, Any
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama
from github import Github, Auth
from contextlib import asynccontextmanager

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

# Global handles
tunnel_process: Optional[subprocess.Popen] = None
MY_TUNNEL_URL: Optional[str] = None

# ---------------------------------------------------------------------------
# Cloudflare Tunnel Manager
# ---------------------------------------------------------------------------
def start_cloudflare_tunnel() -> Optional[str]:
    global tunnel_process
    cmd: str = "./cloudflared" if os.path.exists("./cloudflared") else "cloudflared"
    
    try:
        subprocess.run([cmd, "--version"], capture_output=True, check=True)
    except Exception as e:
        print(f"cloudflared binary not found or not working: {e}. Running without tunnel.", flush=True)
        return None

    print(f"Starting cloudflared tunnel using: {cmd}", flush=True)
    try:
        log_file = open("tunnel.log", "w")
        tunnel_process = subprocess.Popen(
            [cmd, "tunnel", "--url", "http://localhost:8000"],
            stdout=log_file,
            stderr=subprocess.STDOUT
        )
        
        url: Optional[str] = None
        for i in range(15):
            time.sleep(1)
            if os.path.exists("tunnel.log"):
                with open("tunnel.log", "r") as f:
                    content: str = f.read()
                    match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", content)
                    if match:
                        url = match.group(0)
                        break
        log_file.close()
        return url
    except Exception as ex:
        print(f"Failed to start cloudflared tunnel process: {ex}", flush=True)
        return None

# ---------------------------------------------------------------------------
# GitHub DNS Updater (Via Cloudflare Worker)
# ---------------------------------------------------------------------------
def update_github_dns(pat: str, org: str, public_url: str, repo_name: str) -> None:
    max_attempts: int = 5
    match = re.match(r"^(.*?)-(\d+)$", repo_name)
    superkey = match.group(1) if match else "kv"
    
    dns_key = f"{superkey}/{repo_name}"
    print(f"Updating DNS registry via Cloudflare Worker... Key: {dns_key}", flush=True)
    
    for attempt in range(1, max_attempts + 1):
        try:
            payload = {"key": dns_key, "value": public_url}
            res = requests.post("https://dns-manager.aakashmishra2050880.workers.dev/update", json=payload, timeout=10)
            if res.status_code == 200:
                print(f"DNS updated successfully for key '{dns_key}' with URL {public_url}", flush=True)
                return
            else:
                print(f"CF Worker returned status code {res.status_code}: {res.text}", flush=True)
        except Exception as e:
            import random
            print(f"Error updating DNS (attempt {attempt}/{max_attempts}): {e}", flush=True)
            time.sleep(random.uniform(2.0, 5.0))

def trigger_self_workflow(pat: str, org: str, repo_name: str) -> None:
    print(f"Triggering self workflow dispatch for repository {repo_name}...", flush=True)
    try:
        auth_obj: Auth.Token = Auth.Token(pat)
        g: Github = Github(auth=auth_obj)
        repo = g.get_repo(f"{org}/{repo_name}")
        default_branch: str = repo.default_branch
        
        # Trigger standard workflow.yml on the default branch
        wf = repo.get_workflow("workflow.yml")
        wf.create_dispatch(default_branch)
        print("Self workflow dispatch triggered successfully.", flush=True)
    except Exception as e:
        print(f"Failed to trigger self workflow: {e}", flush=True)

def shutdown_timer(pat: str, org: str, repo_name: str, duration_hours: float) -> None:
    duration_seconds: float = duration_hours * 3600
    print(f"Graceful shutdown timer started: Server will run for {duration_hours} hours ({duration_seconds} seconds).", flush=True)
    
    time.sleep(duration_seconds)
    
    print("Timer expired. Initiating graceful shutdown and restart...", flush=True)
    
    # 1. Trigger next workflow run
    if pat and repo_name != "test":
        trigger_self_workflow(pat, org, repo_name)
        
    time.sleep(5)
    
    # 2. Kill cloudflared tunnel
    global tunnel_process
    if tunnel_process:
        try:
            tunnel_process.terminate()
        except Exception:
            pass
        
    print("Exiting server process gracefully with code 0.", flush=True)
    os._exit(0)

# ---------------------------------------------------------------------------
# Lifespan Events Handler (Startup & Shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MY_TUNNEL_URL
    
    pat: str = os.getenv("GITHUB_PAT", "")
    org: str = os.getenv("GITHUB_ORG", "LeadbaseAI-Official")
    repo_full: str = os.getenv("GITHUB_REPOSITORY", "")
    repo_name: str = repo_full.split("/")[-1] if "/" in repo_full else "test"

    # Start the shutdown timer thread
    duration_str: str = os.getenv("RUN_DURATION_HOURS", "4.0")
    try:
        duration_hours: float = float(duration_str)
    except ValueError:
        duration_hours = 4.0

    t: threading.Thread = threading.Thread(
        target=shutdown_timer,
        args=(pat, org, repo_name, duration_hours),
        daemon=True
    )
    t.start()

    # Pre-warm model weight values
    try:
        get_llm()
    except Exception as e:
        print(f"[Warmup] Warning: model warmup failed: {e}", flush=True)

    # Start Cloudflare Quick Tunnel
    MY_TUNNEL_URL = start_cloudflare_tunnel()
    if MY_TUNNEL_URL:
        print(f"==================================================", flush=True)
        print(f"KV CACHE COMPILER TUNNEL ESTABLISHED SUCCESSFULLY!", flush=True)
        print(f"Address: {MY_TUNNEL_URL}", flush=True)
        print(f"==================================================", flush=True)
        
        # Register itself under the kv-worker DNS key
        if pat:
            update_github_dns(pat, org, MY_TUNNEL_URL, repo_name)
    else:
        print("Running kv worker without public tunnel.", flush=True)
        
    yield

app = FastAPI(title="KV Global Cache Pre-compiler Worker", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Redis Connection Sync
# ---------------------------------------------------------------------------
def upload_to_redis(client_id: str, state_data: bytes) -> bool:
    """
    Looks up the active redis-worker URL from config.json and pushes the global state payload.
    """
    org: str = os.getenv("GITHUB_ORG", "LeadbaseAI-Official")
    print(f"[Redis Link] Fetching live redis-worker endpoint address...", flush=True)
    try:
        res_dns = requests.get(f"https://raw.githubusercontent.com/{org}/dns/main/config.json", timeout=10)
        if res_dns.status_code != 200:
            print(f"[Redis Link] Failed to fetch DNS registry from GitHub raw (Code: {res_dns.status_code})", flush=True)
            return False
            
        config_data = res_dns.json()
        redis_url: Optional[str] = config_data.get("redis", {}).get("redis-worker")
        
        if not redis_url:
            print("[Redis Link] Error: redis-worker URL not registered in DNS registry.", flush=True)
            return False

        b64_str = base64.b64encode(state_data).decode("utf-8")
        payload = {
            "key": f"global:{client_id}",
            "value": b64_str
        }
        
        res = requests.post(f"{redis_url.rstrip('/')}/add", json=payload, timeout=30)
        if res.status_code == 200:
            print(f"[Redis Link] Successfully pushed compiled global prefix for client {client_id} to Redis.", flush=True)
            return True
        else:
            print(f"[Redis Link] Redis endpoint returned status {res.status_code}: {res.text}", flush=True)
            return False
    except Exception as e:
        print(f"[Redis Link] Error linking to redis-worker: {e}", flush=True)
        return False

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.post("/update")
def update_global_cache(req: UpdateRequest) -> Dict[str, Any]:
    try:
        t0 = time.time()
        llm = get_llm()
        
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
        
        tokens = llm.tokenize(stitched_text.encode("utf-8"))
        print(f"[KV Worker] Tokenizing prompt for client {req.client_id} (Token count: {len(tokens)})", flush=True)
        
        llm.reset()
        llm.eval(tokens)
        
        state_obj = llm.save_state()
        state_bytes = pickle.dumps(state_obj)
        
        # Call active upload
        uploaded = upload_to_redis(req.client_id, state_bytes)
        
        duration = time.time() - t0
        return {
            "status": "success" if uploaded else "partial_success_local_only",
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
    # Server runs on port 8000 internally (mapped via cloudflared to localhost:8000)
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False, access_log=False)
