import os
import time
import json
import base64
import pickle
import threading
import subprocess
import re
import requests
import uvicorn
import datetime
import gzip
from typing import Optional, Dict, Any
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama, GGML_TYPE_Q8_0 # type: ignore
from github import Github, Auth
from contextlib import asynccontextmanager

class UpdateRequest(BaseModel):
    client_id: str
    system_prompt: str
    persona: str
    kb: str

# Standardized logging helper: [HH:MM:SS | DD] [tag] : msg
def log_message(tag: str, msg: str) -> None:
    from datetime import datetime as dt, timezone, timedelta
    ist_now = dt.now(timezone.utc) + timedelta(hours=5, minutes=30)
    now_str = ist_now.strftime("%H:%M:%S")
    day_str = ist_now.strftime("%d")
    print(f"[{now_str} | {day_str}] [{tag}] : {msg}", flush=True)

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
                log_message("system", f"Loading model weights from {model_path}...")
                _llm = Llama(
                    model_path=str(model_path),
                    n_ctx=40960,
                    n_threads=2,
                    flash_attn=True,
                    type_k=GGML_TYPE_Q8_0,
                    type_v=GGML_TYPE_Q8_0
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
        log_message("system", f"cloudflared binary not found or not working: {e}. Running without tunnel.")
        return None

    log_message("system", f"Starting cloudflared tunnel using: {cmd}")
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
        log_message("system", f"Failed to start cloudflared tunnel process: {ex}")
        return None

SUPERKEY = "kv-worker"

# ---------------------------------------------------------------------------
# GitHub DNS Updater (Via Cloudflare Worker)
# ---------------------------------------------------------------------------
def update_github_dns(pat: str, org: str, public_url: str, repo_name: str) -> None:
    max_attempts: int = 5
    dns_key = f"{SUPERKEY}/{repo_name}"
    log_message("system", f"Updating DNS registry via Cloudflare Worker... Key: {dns_key}")
    
    for attempt in range(1, max_attempts + 1):
        try:
            payload = {"key": dns_key, "value": public_url}
            res = requests.post("https://dns-manager.aakashmishra2050880.workers.dev/update", json=payload, timeout=10)
            if res.status_code == 200:
                log_message("system", f"DNS updated successfully for key '{dns_key}' with URL {public_url}")
                return
            else:
                log_message("system", f"CF Worker returned status code {res.status_code}: {res.text}")
        except Exception as e:
            import random
            log_message("system", f"Error updating DNS (attempt {attempt}/{max_attempts}): {e}")
            time.sleep(random.uniform(2.0, 5.0))

def trigger_self_workflow(pat: str, org: str, repo_name: str) -> None:
    log_message("system", f"Triggering self workflow dispatch for repository {repo_name}...")
    try:
        auth_obj: Auth.Token = Auth.Token(pat)
        g: Github = Github(auth=auth_obj)
        repo = g.get_repo(f"{org}/{repo_name}")
        default_branch: str = repo.default_branch
        
        wf = repo.get_workflow("workflow.yml")
        wf.create_dispatch(default_branch)
        log_message("system", "Self workflow dispatch triggered successfully.")
    except Exception as e:
        log_message("system", f"Failed to trigger self workflow: {e}")

def shutdown_timer(pat: str, org: str, repo_name: str, duration_hours: float) -> None:
    duration_seconds: float = duration_hours * 3600
    log_message("system", f"Graceful shutdown timer started: Server will run for {duration_hours} hours ({duration_seconds} seconds).")
    
    time.sleep(duration_seconds)
    
    log_message("system", "Timer expired. Initiating graceful shutdown and restart...")
    
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
        
    log_message("system", "Exiting server process gracefully with code 0.")
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
        log_message("system", f"Warning: model warmup failed: {e}")

    # Start Cloudflare Quick Tunnel
    MY_TUNNEL_URL = start_cloudflare_tunnel()
    if MY_TUNNEL_URL:
        log_message("system", f"KV CACHE COMPILER TUNNEL ESTABLISHED SUCCESSFULLY! Address: {MY_TUNNEL_URL}")
        
        # Register itself under the kv-worker DNS key
        if pat:
            update_github_dns(pat, org, MY_TUNNEL_URL, repo_name)
    else:
        log_message("system", "Running kv worker without public tunnel.")
        
    yield

app = FastAPI(title="KV Global Cache Pre-compiler Worker", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Redis Connection Sync
# ---------------------------------------------------------------------------
def upload_to_redis(client_id: str, state_data: bytes) -> bool:
    org: str = os.getenv("GITHUB_ORG", "LeadbaseAI-Official")
    
    import gzip
    compressed_data = gzip.compress(state_data)
    log_message("system", f"Compressed state from {len(state_data)} to {len(compressed_data)} bytes.")
    b64_str = base64.b64encode(compressed_data).decode("utf-8")
    
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        log_message("system", f"Fetching live redis-worker endpoint address (Attempt {attempt}/{max_attempts})...")
        try:
            # Query registry with a dynamic timestamp parameter to bypass raw GitHub CDN caching
            timestamp = int(time.time())
            res_dns = requests.get(f"https://raw.githubusercontent.com/{org}/dns/main/config.json?t={timestamp}", timeout=10)
            if res_dns.status_code != 200:
                log_message("system", f"Failed to fetch DNS registry from GitHub raw (Code: {res_dns.status_code})")
                time.sleep(10)
                continue
                
            config_data = res_dns.json()
            redis_url: Optional[str] = config_data.get("redis-worker", {}).get("active")

            if not redis_url:
                log_message("system", "No active redis-worker URL registered. Falling back to direct broadcast to active runners...")
                targets = []
                for category, sub_dict in config_data.items():
                    if category in ("standby-server", "redis-worker", "kv-worker", "standby", "redis"):
                        continue
                    if isinstance(sub_dict, dict):
                        for runner_name, url in sub_dict.items():
                            if url and url.startswith("https://"):
                                targets.append((runner_name, url))
                
                payload = {
                    "client_id": client_id,
                    "state_bytes_base64": b64_str
                }
                
                success_count = 0
                for name, url in targets:
                    try:
                        res = requests.post(f"{url.rstrip('/')}/v1/global-update", json=payload, timeout=15)
                        if res.status_code == 200:
                            log_message("system", f"Successfully synced client cache directly to runner: {name}")
                            success_count += 1
                        else:
                            log_message("system", f"Runner {name} returned status {res.status_code}")
                    except Exception as ex:
                        log_message("system", f"Error syncing directly to runner {name}: {ex}")
                
                if success_count > 0:
                    return True
                time.sleep(10)
                continue

            payload = {
                "key": f"global:{client_id}",
                "value": b64_str
            }
            
            res = requests.post(f"{redis_url.rstrip('/')}/add", json=payload, timeout=30)
            if res.status_code == 200:
                log_message("system", f"Successfully pushed compiled global prefix for client {client_id} to Redis.")
                return True
            else:
                log_message("system", f"Redis endpoint returned status {res.status_code}. Retrying...")
        except Exception as e:
            log_message("system", f"Error linking to redis-worker (Attempt {attempt}/{max_attempts}): {e}")
            
        if attempt < max_attempts:
            time.sleep(10)
            
    log_message("system", f"Failed to push compiled prefix to Redis after {max_attempts} attempts.")
    return False
_eval_lock = threading.Lock()

class SummarizeRequest(BaseModel):
    client_id: str
    phone_number: str
    history: list[dict[str, str]]
    system_prompt: str
    persona: str
    kb: str

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
        system_content = "\n".join(prompt_parts)
        stitched_text = f"<|im_start|>system\n{system_content}<|im_end|>\n"
        
        tokens = llm.tokenize(stitched_text.encode("utf-8"))
        log_message("system", f"Tokenizing prompt for client {req.client_id} (Token count: {len(tokens)})")
        
        with _eval_lock:
            llm.reset()
            llm.eval(tokens)
            state_obj = llm.save_state()
        
        # Save both state and the token sequence that generated it
        payload_obj = {
            "state": state_obj,
            "tokens": tokens
        }
        state_bytes = pickle.dumps(payload_obj)
        
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

_summary_jobs: Dict[str, Dict[str, Any]] = {}
_summary_jobs_lock = threading.Lock()

def _process_summary_job(job_id: str, req: SummarizeRequest) -> None:
    try:
        t0 = time.time()
        llm = get_llm()
        
        history_lines = []
        for msg in req.history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            history_lines.append(f"{role.capitalize()}: {content}")
        history_text = "\n".join(history_lines)
        
        summary_prompt = (
            f"<|im_start|>system\n"
            f"You are a helpful assistant that summarizes customer service chat transcripts concisely.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Summarize the following chat transcript in 5 concise bullet points. Highlight the customer's questions, interests, and needs:\n"
            f"{history_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        tokens_prompt = llm.tokenize(summary_prompt.encode("utf-8"))
        
        with _eval_lock:
            llm.reset()
            res = llm.create_completion(
                prompt=tokens_prompt,
                max_tokens=150,
                temperature=0.3
            )
            summary_text = res["choices"][0]["text"].strip()
            
            if "<think>" in summary_text:
                summary_text = re.sub(r"<think>.*?</think>", "", summary_text, flags=re.DOTALL).strip()
            
            log_message("system", f"[Job {job_id[:8]}] Generated conversation summary for {req.phone_number}: {summary_text[:100]}...")
            
            prompt_parts = [
                "System Prompt:",
                req.system_prompt.strip(),
                "",
                "Persona:",
                req.persona.strip(),
                "",
                "Knowledge Base (Authoritative Facts):",
                req.kb.strip(),
                "",
                "Summary of previous conversation:",
                summary_text,
                ""
            ]
            system_content = "\n".join(prompt_parts)
            stitched_text = f"<|im_start|>system\n{system_content}<|im_end|>\n"
            
            tokens = llm.tokenize(stitched_text.encode("utf-8"))
            llm.reset()
            llm.eval(tokens)
            state_obj = llm.save_state()
            
        payload_obj = {
            "state": state_obj,
            "tokens": tokens
        }
        state_bytes = pickle.dumps(payload_obj)
        state_bytes_base64 = base64.b64encode(state_bytes).decode("utf-8")
        
        duration = time.time() - t0
        with _summary_jobs_lock:
            _summary_jobs[job_id] = {
                "status": "completed",
                "summary": summary_text,
                "state_bytes_base64": state_bytes_base64,
                "tokens_compiled": len(tokens),
                "compilation_time_seconds": round(duration, 3)
            }
        log_message("system", f"[Job {job_id[:8]}] Summarization job completed in {round(duration, 2)}s for {req.phone_number}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        with _summary_jobs_lock:
            _summary_jobs[job_id] = {
                "status": "failed",
                "error": str(e)
            }
        log_message("system", f"[Job {job_id[:8]}] Summarization job failed: {e}")

@app.post("/summarize")
def summarize_history(req: SummarizeRequest) -> Dict[str, Any]:
    import uuid
    job_id = str(uuid.uuid4())
    with _summary_jobs_lock:
        _summary_jobs[job_id] = {"status": "pending"}
    
    t = threading.Thread(target=_process_summary_job, args=(job_id, req), daemon=True)
    t.start()
    
    log_message("system", f"Dispatched background summary job {job_id[:8]} for phone {req.phone_number}")
    return {
        "status": "pending",
        "job_id": job_id
    }

@app.get("/summarize/status")
def get_summary_status(job_id: str) -> Dict[str, Any]:
    with _summary_jobs_lock:
        job = _summary_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return job

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False, access_log=False)
