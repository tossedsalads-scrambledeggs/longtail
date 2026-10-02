"""Longtail web app: find and AI-verify a scenario across every camera.

Runs behind an Ingress that strips the /app/longtail prefix, so routes live at /.
Reads VSS_URL, VSS_USERNAME, VSS_PASSWORD, WANDB_API_KEY, WANDB_TEAM, WANDB_PROJECT
and PORT from the environment.
"""
import logging
import os
import threading
import uuid
from collections import OrderedDict
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from longtail import VSS, diagnose_misses, enable_weave, find_scenario, judge_clip

logging.getLogger("httpx").setLevel(logging.WARNING)

HERE = Path(__file__).parent
MAX_JOBS = 20
STREAM_HEADERS = ("content-length", "content-range", "accept-ranges")

app = FastAPI()
judge, diagnose, weave_status = enable_weave(judge_clip, diagnose_misses)
jobs = OrderedDict()
jobs_lock = threading.Lock()
run_lock = threading.Lock()
vss_holder = {}
http = httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=120.0))


def new_vss():
    vss = VSS(os.environ["VSS_URL"], os.environ["VSS_USERNAME"], os.environ["VSS_PASSWORD"])
    vss_holder["vss"] = vss
    return vss


def public(hit):
    return {k: v for k, v in hit.items() if k != "source"}


def run_job(job_id, scenario):
    job = jobs[job_id]

    def record(event):
        if event["kind"] == "searched":
            job["hits"] = event["hits"]
            return
        if event["kind"] == "verdict":
            event = {**event, "hit": public(job["hits"][event["clip"]])}
        job["events"].append(event)

    try:
        result = find_scenario(new_vss(), scenario, judge, diagnose, on_event=record)
        job.update(status="done", **result)
    except (Exception, SystemExit) as e:
        job.update(status="error", error=str(e)[:300])
    finally:
        run_lock.release()


class RunRequest(BaseModel):
    scenario: str


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/health")
def health():
    return {"ok": True, "weave": weave_status}


@app.post("/api/run")
def start_run(req: RunRequest):
    scenario = " ".join(req.scenario.split())
    if not scenario:
        raise HTTPException(400, "Enter a scenario.")
    if not run_lock.acquire(blocking=False):
        raise HTTPException(409, "Another search is already running. Try again in a minute.")
    job_id = uuid.uuid4().hex[:12]
    with jobs_lock:
        jobs[job_id] = {"status": "running", "scenario": scenario, "events": []}
        while len(jobs) > MAX_JOBS:
            jobs.popitem(last=False)
    threading.Thread(target=run_job, args=(job_id, scenario), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job.")
    out = {"status": job["status"], "scenario": job["scenario"], "events": list(job["events"])}
    if job["status"] == "error":
        out["error"] = job["error"]
    if job["status"] == "done":
        out.update(
            cameras=job["cameras"],
            empty_cameras=job["empty_cameras"],
            hits=[public(hit) for hit in job["hits"]],
            insight=job["insight"],
        )
    return out


@app.get("/clip/{job_id}/{index}")
async def clip(job_id: str, index: int, request: Request):
    job = jobs.get(job_id)
    hits = job.get("hits") if job else None
    if not hits or not 0 <= index < len(hits):
        raise HTTPException(404, "Unknown clip.")
    source = hits[index]["source"]
    headers = {"Range": request.headers["range"]} if "range" in request.headers else {}

    for attempt in range(2):
        vss = vss_holder.get("vss") or await run_in_threadpool(new_vss)
        upstream = await http.send(
            http.build_request(
                "GET", f"{vss.backend}/api/v1/videos/stream",
                params={"source": source, "token": vss.token}, headers=headers,
            ),
            stream=True,
        )
        if upstream.status_code != 401 or attempt:
            break
        await upstream.aclose()
        await run_in_threadpool(new_vss)

    if upstream.status_code >= 400:
        await upstream.aclose()
        raise HTTPException(502, f"Clip unavailable (backend returned {upstream.status_code}).")
    return StreamingResponse(
        upstream.aiter_raw(),
        status_code=upstream.status_code,
        media_type="video/mp4",
        headers={k: upstream.headers[k] for k in STREAM_HEADERS if k in upstream.headers},
        background=BackgroundTask(upstream.aclose),
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
