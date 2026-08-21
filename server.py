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
from typing import Optional, Dict, Any, List

from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama, GGML_TYPE_Q8_0 # type: ignore
from github import Github, Auth
from contextlib import asynccontextmanager

class UpdateRequest(BaseModel):
    client_id: str
    model_id: Optional[str] = "0bm-1"
    system_prompt: str
    persona: str
    kb: str


def log_message(tag: str, msg: str) -> None:
    from datetime import datetime as dt, timezone, timedelta
    ist_now = dt.now(timezone.utc) + timedelta(hours=5, minutes=30)
    now_str = ist_now.strftime("%H:%M:%S")
    day_str = ist_now.strftime("%d")
    print(f"[{now_str} | {day_str}] [{tag}] : {msg}", flush=True)

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

tunnel_process: Optional[subprocess.Popen] = None
MY_TUNNEL_URL: Optional[str] = None

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
        for _ in range(15):
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
    log_message("system", f"Graceful shutdown timer started: Server will run for {duration_hours} hours.")
    time.sleep(duration_seconds)
    log_message("system", "Timer expired. Initiating graceful shutdown and restart...")
    if pat and repo_name != "test":
        trigger_self_workflow(pat, org, repo_name)
    time.sleep(5)
    global tunnel_process
    if tunnel_process:
        try:
            tunnel_process.terminate()
        except Exception:
            pass
    log_message("system", "Exiting server process gracefully with code 0.")
    os._exit(0)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MY_TUNNEL_URL
    pat: str = os.getenv("GITHUB_PAT", "")
    org: str = os.getenv("GITHUB_ORG", "LeadbaseAI-Official")
    repo_full: str = os.getenv("GITHUB_REPOSITORY", "")
    repo_name: str = repo_full.split("/")[-1] if "/" in repo_full else "test"

    duration_str: str = os.getenv("RUN_DURATION_HOURS", "4.0")
    try:
        duration_hours: float = float(duration_str)
    except ValueError:
        duration_hours = 4.0

    threading.Thread(
        target=shutdown_timer,
        args=(pat, org, repo_name, duration_hours),
        daemon=True
    ).start()

    try:
        get_llm()
    except Exception as e:
        log_message("system", f"Warning: model warmup failed: {e}")

    MY_TUNNEL_URL = start_cloudflare_tunnel()
    if MY_TUNNEL_URL:
        log_message("system", f"KV CACHE COMPILER TUNNEL ESTABLISHED! Address: {MY_TUNNEL_URL}")
        if pat:
            update_github_dns(pat, org, MY_TUNNEL_URL, repo_name)
    else:
        log_message("system", "Running kv worker without public tunnel.")
    yield

app = FastAPI(title="KV Global Cache Pre-compiler Worker", lifespan=lifespan)

_pending_retries: List[Dict[str, Any]] = []
_retry_queue_lock: threading.Lock = threading.Lock()

def fetch_fresh_dns_config() -> Optional[Dict[str, Any]]:
    """
    Fetches fresh DNS registry config directly from GitHub API with cache bypassing.
    """
    org: str = os.getenv("GITHUB_ORG", "LeadbaseAI-Official")
    pat: str = os.getenv("GITHUB_PAT", "")
    headers: Dict[str, str] = {
        "User-Agent": "LeadBaseAI-Worker",
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache"
    }
    if pat:
        headers["Authorization"] = f"token {pat}"
        
    try:
        api_url: str = f"https://api.github.com/repos/{org}/dns/contents/config.json"
        res_api = requests.get(api_url, headers=headers, timeout=6)
        if res_api.status_code == 200:
            api_json: Dict[str, Any] = res_api.json()
            if "content" in api_json:
                decoded: str = base64.b64decode(api_json["content"]).decode("utf-8")
                return json.loads(decoded)
    except Exception as e:
        log_message("dns", f"GitHub API DNS fetch failed: {e}")
        
    try:
        timestamp: int = int(time.time())
        raw_url: str = f"https://raw.githubusercontent.com/{org}/dns/main/config.json?t={timestamp}"
        res_dns = requests.get(raw_url, headers=headers, timeout=6)
        if res_dns.status_code == 200:
            return res_dns.json()
    except Exception as e:
        log_message("dns", f"Raw GitHub DNS fetch failed: {e}")
        
    return None

def resolve_runner_url_from_dns(model_id: str) -> Optional[str]:
    """
    Resolves target model runner active tunnel URL from DNS registry.
    """
    config_data: Optional[Dict[str, Any]] = fetch_fresh_dns_config()
    if not config_data:
        return None
        
    for category, val in config_data.items():
        if isinstance(val, dict) and model_id in val:
            url: str = str(val[model_id])
            if url and url.startswith("http"):
                return url
        elif isinstance(val, str) and category == model_id:
            if val.startswith("http"):
                return val
                
    return None

def send_state_to_model_runner(model_id: str, client_id: str, b64_str: str) -> bool:
    """
    Attempts to push compiled KV state directly to the targeted model runner via a fresh HTTP connection.
    """
    runner_url: Optional[str] = resolve_runner_url_from_dns(model_id)
    if not runner_url:
        log_message("SYNC", f"❌ Target runner '{model_id}' URL not found in DNS registry (runner may be booting).")
        return False
        
    endpoint: str = f"{runner_url.rstrip('/')}/v1/global-update"
    payload: Dict[str, Any] = {
        "client_id": client_id,
        "state_bytes_base64": b64_str
    }
    
    try:
        log_message("SYNC", f"🚀 Dispatching compiled KV state to target model runner '{model_id}' at {endpoint}...")
        res = requests.post(endpoint, json=payload, timeout=20)
        if res.status_code == 200:
            log_message("SYNC", f"✅ SUCCESS: KV state delivered to model runner '{model_id}'!")
            return True
        else:
            log_message("SYNC", f"⚠️ Model runner '{model_id}' returned HTTP {res.status_code}: {res.text}")
            return False
    except Exception as err:
        log_message("SYNC", f"❌ Connection error sending state to model runner '{model_id}': {err}")
        return False

def enqueue_for_retry(model_id: str, client_id: str, b64_str: str) -> None:
    """
    Enqueues compiled state in memory for background retry (every 2 minutes).
    """
    with _retry_queue_lock:
        item: Dict[str, Any] = {
            "model_id": model_id,
            "client_id": client_id,
            "b64_str": b64_str,
            "attempts": 0,
            "next_retry_timestamp": time.time() + 120
        }
        _pending_retries.append(item)
        log_message("RETRY_QUEUE", f"📦 Enqueued compiled state for client='{client_id}', model='{model_id}' in 2-min memory retry queue.")

def _kv_retry_worker() -> None:
    """
    Background worker thread continuously processing memory retries every 2 minutes with fresh DNS lookups.
    """
    while True:
        time.sleep(10)
        now: float = time.time()
        to_retry: List[Dict[str, Any]] = []
        
        with _retry_queue_lock:
            remaining: List[Dict[str, Any]] = []
            for item in _pending_retries:
                if now >= item["next_retry_timestamp"]:
                    to_retry.append(item)
                else:
                    remaining.append(item)
            _pending_retries.clear()
            _pending_retries.extend(remaining)
            
        for item in to_retry:
            m_id: str = item["model_id"]
            c_id: str = item["client_id"]
            b64: str = item["b64_str"]
            attempts: int = item["attempts"] + 1
            
            log_message("RETRY_WORKER", f"[Retry Attempt #{attempts}] Re-evaluating fresh DNS for runner '{m_id}' (client='{c_id}')...")
            success: bool = send_state_to_model_runner(m_id, c_id, b64)
            if success:
                log_message("RETRY_WORKER", f"✅ Retry attempt #{attempts} succeeded for client='{c_id}' -> runner '{m_id}'!")
            else:
                item["attempts"] = attempts
                item["next_retry_timestamp"] = time.time() + 120
                with _retry_queue_lock:
                    _pending_retries.append(item)
                log_message("RETRY_WORKER", f"❌ Retry attempt #{attempts} failed for runner '{m_id}'. Will retry again in 2 minutes.")

# Start background retry worker thread on module load
threading.Thread(target=_kv_retry_worker, daemon=True).start()

def sync_kv_to_target_runner(model_id: str, client_id: str, state_bytes: bytes) -> Dict[str, Any]:
    """
    Compresses state bytes and pushes to target model runner. Enqueues in memory if runner is offline/booting.
    """
    compressed_data: bytes = gzip.compress(state_bytes)
    log_message("COMPRESSION", f"State compressed from {len(state_bytes)} to {len(compressed_data)} bytes (~{len(compressed_data)/(1024*1024):.2f} MB).")
    b64_str: str = base64.b64encode(compressed_data).decode("utf-8")
    
    delivered: bool = send_state_to_model_runner(model_id, client_id, b64_str)
    if delivered:
        log_message("RESULT", f"State delivery COMPLETE for client='{client_id}' -> model='{model_id}'")
        return {"status": "delivered", "model_id": model_id, "client_id": client_id}
    else:
        enqueue_for_retry(model_id, client_id, b64_str)
        log_message("RESULT", f"Target runner '{model_id}' offline/booting -> Enqueued for 2-minute memory retries.")
        return {"status": "enqueued_for_retry", "model_id": model_id, "client_id": client_id}



class CustomerSummaryCompileRequest(BaseModel):
    model_id: str = "0bm-1"
    customer_key: str
    system_prompt: str
    summary: str
    upload_hf: bool = True

HF_REPO_ID = "anisoleai/client-states"

def upload_customer_state_to_hf(model_id: str, customer_key: str, state_bytes: bytes) -> bool:
    """
    Uploads a pre-compiled customer state binary directly to Hugging Face private dataset repo.
    Returns True if successfully uploaded, False otherwise.
    """
    token: str = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("GITHUB_PAT") or ""
    if not token:
        log_message("system", "[HF Upload] No HF_TOKEN provided. Skipping Hugging Face upload.")
        return False

    log_message("system", f"[HF Upload] Syncing compiled state for customer '{customer_key}' under model '{model_id}'...")
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        path_in_repo = f"models/{model_id}/states/{customer_key}.bin"
        
        api.upload_file(
            path_or_fileobj=state_bytes,
            path_in_repo=path_in_repo,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"Upload pre-compiled summary KV state for customer {customer_key}"
        )
        log_message("system", f"[HF Upload] Successfully uploaded state to '{path_in_repo}'!")
        return True
    except Exception as err:
        log_message("system", f"[HF Upload] Failed to upload customer state to HF: {err}")
        return False

_eval_lock = threading.Lock()

@app.post("/compile-summary-kv")
def compile_summary_kv(req: CustomerSummaryCompileRequest) -> Dict[str, Any]:
    """
    Pre-compiles KV cache for (System Prompt + Conversation Summary) WITHOUT generating any text answer.
    Exports compact KV state binary and uploads directly to Hugging Face dataset.
    """
    try:
        t0 = time.time()
        log_message("debug", f"[summary-compile] : Received request for customer_key={req.customer_key}, model_id={req.model_id}")
        
        # Format Qwen ChatML system prompt and context summary
        prompt_text = (
            f"<|im_start|>system\n{req.system_prompt.strip()}<|im_end|>\n"
            f"<|im_start|>user\n[Context Summary of Past Conversations]:\n{req.summary.strip()}<|im_end|>\n"
            f"<|im_start|>assistant\nUnderstood. I have loaded the context summary.<|im_end|>\n"
        )
        
        base_llm = get_llm()
        tokens = base_llm.tokenize(prompt_text.encode("utf-8"))
        log_message("debug", f"[summary-compile] : Tokenized system + summary prompt (Token count: {len(tokens)})")
        
        # DYNAMIC CONTEXT EVALUATION ONLY (No generation inference!)
        model_path = find_gguf_file()
        with _eval_lock:
            comp_llm = Llama(
                model_path=str(model_path),
                n_ctx=len(tokens) + 64,
                n_threads=2,
                flash_attn=True,
                type_k=GGML_TYPE_Q8_0,
                type_v=GGML_TYPE_Q8_0
            )
            # Pre-populate Key/Value matrices in memory without text generation
            comp_llm.eval(tokens)
            state_obj = comp_llm.save_state()
            del comp_llm  # Free temporary compiler instance
        
        customer_obj = {
            "customer_key": req.customer_key,
            "state": state_obj,
            "tokens": tokens,
            "history": [
                {"role": "user", "content": f"[Context Summary]\n{req.summary}"},
                {"role": "assistant", "content": "Understood. I have loaded the context summary."}
            ],
            "msg_count": 2
        }
        state_bytes = pickle.dumps(customer_obj)
        log_message("debug", f"[summary-compile] : State size: {len(state_bytes)} bytes (~{len(state_bytes) / (1024*1024):.1f} MB)")
        
        hf_success: bool = False
        if req.upload_hf:
            hf_success = upload_customer_state_to_hf(req.model_id, req.customer_key, state_bytes)
            
        sync_res: Dict[str, Any] = sync_kv_to_target_runner(req.model_id, req.customer_key, state_bytes)
            
        duration: float = time.time() - t0
        return {
            "status": sync_res.get("status", "success"),
            "model_id": req.model_id,
            "customer_key": req.customer_key,
            "tokens_compiled": len(tokens),
            "compilation_time_seconds": round(duration, 3),
            "state_size_bytes": len(state_bytes),
            "hf_uploaded": hf_success,
            "runner_delivery": sync_res.get("status")
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Summary KV Compilation failed: {str(e)}")

@app.post("/update")
def update_global_cache(req: UpdateRequest) -> Dict[str, Any]:
    try:
        t0 = time.time()
        target_model: str = req.model_id or "0bm-1"
        
        print("\n" + "═" * 65, flush=True)
        log_message("UPDATE_REQUEST", f"📥 RECEIVED COMPILATION REQUEST: client_id='{req.client_id}', target_model='{target_model}'")
        log_message("UPDATE_REQUEST", f"   System Prompt : {len(req.system_prompt)} chars")
        log_message("UPDATE_REQUEST", f"   Persona       : {len(req.persona)} chars")
        log_message("UPDATE_REQUEST", f"   KnowledgeBase : {len(req.kb)} chars")
        print("═" * 65, flush=True)
        
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
        
        # Tokenize using baseline model
        base_llm = get_llm()
        tokens = base_llm.tokenize(stitched_text.encode("utf-8"))
        log_message("TOKENIZER", f"Tokenizer generated {len(tokens)} tokens for client_id='{req.client_id}'")
        
        # DYNAMIC CONTEXT COMPRESSION:
        # Create a dynamically-sized Llama instance (n_ctx = len(tokens) + 64)
        # Exports a compact ~34.5 MB state file instead of 564 MB!
        model_path = find_gguf_file()
        with _eval_lock:
            comp_llm = Llama(
                model_path=str(model_path),
                n_ctx=len(tokens) + 64,
                n_threads=2,
                flash_attn=True,
                type_k=GGML_TYPE_Q8_0,
                type_v=GGML_TYPE_Q8_0
            )
            comp_llm.eval(tokens)
            state_obj = comp_llm.save_state()
            del comp_llm  # Free temporary compiler instance
        
        payload_obj = {
            "state": state_obj,
            "tokens": tokens
        }
        state_bytes = pickle.dumps(payload_obj)
        log_message("COMPILER", f"Compiled binary KV state size: {len(state_bytes)} bytes (~{len(state_bytes) / (1024*1024):.2f} MB)")
        
        sync_result: Dict[str, Any] = sync_kv_to_target_runner(target_model, req.client_id, state_bytes)
        
        duration = time.time() - t0
        log_message("UPDATE_COMPLETE", f"Total compilation & dispatch time: {round(duration, 3)} seconds")
        print("═" * 65 + "\n", flush=True)
        
        return {
            "status": sync_result.get("status", "success"),
            "client_id": req.client_id,
            "model_id": target_model,
            "tokens_compiled": len(tokens),
            "compilation_time_seconds": round(duration, 3),
            "state_size_bytes": len(state_bytes)
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Compilation failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False, access_log=False)


