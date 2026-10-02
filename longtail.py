#!/usr/bin/env python3
"""Find and verify a scenario across every camera in the VSS archive.

For each camera, runs a VSS search for the scenario and keeps the top hits, then asks a
W&B Inference model whether each clip really shows the scenario. Confirmed clips are
saved as JSON lines.

Usage:
    python3 tools/longtail/longtail.py "person close to a moving vehicle"
    python3 tools/longtail/longtail.py --per-camera 5 --out found.jsonl forklift near a pedestrian

Reads INGRESS_URL, USERNAME, PASSWORD, WANDB_API_KEY, WANDB_TEAM and WANDB_PROJECT from
the environment. If the weave package is installed, model calls are traced with weave.
"""
import argparse
import json
import os
import re
import sys
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

WANDB_URL = "https://api.inference.wandb.ai/v1/chat/completions"
JUDGE_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B"
JUDGE_MAX_TOKENS = 4096
VERDICTS = ("yes", "no", "unsure")

JUDGE_SYSTEM = (
    "You verify whether a video clip shows a given scenario, using only the clip's text "
    "description and its detected objects. Be strict: answer yes only if the description "
    "clearly shows the scenario, no if it clearly does not, unsure otherwise. "
    "When the scenario involves a person and a vehicle, the person must be outside the "
    "vehicle and on foot (walking, standing, running or working beside it). A person "
    "riding, driving or sitting on a vehicle, including a motorcycle, bicycle, scooter or "
    "forklift, does not count as a person close to that vehicle. Respond with JSON only, "
    'no other text: {"match": "yes" | "no" | "unsure", "reason": "<one short sentence>"}'
)


def env(name):
    value = os.environ.get(name, "")
    if not value:
        sys.exit(f"missing environment variable {name}")
    return value


def request_json(url, payload=None, headers=None, timeout=120):
    data = json.dumps(payload).encode() if payload is not None else None
    # W&B Inference sits behind Cloudflare, which rejects urllib's default User-Agent (error 1010).
    base = {"Content-Type": "application/json", "User-Agent": "longtail/1.0"}
    req = urllib.request.Request(url, data=data, headers={**base, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{url} -> HTTP {e.code}: {e.read().decode(errors='replace')[:300]}") from None


class VSS:
    def __init__(self, backend, username, password):
        self.backend = backend.rstrip("/")
        self._creds = {"username": username, "password": password}
        self.login()

    def login(self):
        self.token = request_json(f"{self.backend}/api/v1/auth/login", self._creds)["access_token"]

    @property
    def auth(self):
        return {"Authorization": f"Bearer {self.token}"}

    def cameras(self):
        query = urllib.parse.urlencode({"field": "camera_id", "limit": 500})
        return request_json(f"{self.backend}/api/v1/metadata/values?{query}", headers=self.auth).get("values", [])

    def search(self, scenario, camera_id, top_k, attempts=3):
        body = {"query": scenario, "top_k": top_k, "llm_top_n": 1, "metadata_filters": {"camera_id": camera_id}}
        for attempt in range(1, attempts + 1):
            try:
                results = request_json(f"{self.backend}/api/v1/search", body, headers=self.auth).get("results", [])
                return results[:top_k]
            except RuntimeError as e:
                if attempt == attempts or not re.search(r"HTTP 50[234]", str(e)):
                    raise
                time.sleep(10 * attempt)


def detected_objects(hit):
    try:
        counts = json.loads(hit.get("object_counts") or "{}")
    except json.JSONDecodeError:
        counts = {}
    if counts:
        return ", ".join(f"{label} x{n}" for label, n in sorted(counts.items(), key=lambda kv: -kv[1]))
    return hit.get("object_classes") or "none"


def parse_verdict(content):
    match = re.search(r"\{.*\}", content or "", re.DOTALL)
    if not match:
        return {"match": "unsure", "reason": "Model returned no JSON verdict."}
    try:
        verdict = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"match": "unsure", "reason": "Model returned malformed JSON."}
    label = str(verdict.get("match", "")).strip().lower()
    return {
        "match": label if label in VERDICTS else "unsure",
        "reason": str(verdict.get("reason", "")).strip() or "No reason given.",
    }


def judge_clip(scenario: str, description: str, objects: str) -> dict:
    headers = {
        "Authorization": f"Bearer {env('WANDB_API_KEY')}",
        "OpenAI-Project": f"{env('WANDB_TEAM')}/{env('WANDB_PROJECT')}",
    }
    body = {
        "model": JUDGE_MODEL,
        "max_tokens": JUDGE_MAX_TOKENS,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": f"Scenario: {scenario}\n\nClip description: {description}\n\nDetected objects: {objects}"},
        ],
    }
    try:
        resp = request_json(WANDB_URL, body, headers=headers, timeout=180)
        return parse_verdict(resp["choices"][0]["message"].get("content"))
    except Exception as e:
        return {"match": "unsure", "reason": f"Model call failed: {str(e)[:120]}"}


def enable_weave(judge):
    os.environ.setdefault("WEAVE_PRINT_CALL_LINK", "false")
    try:
        import weave

        weave.init(f"{env('WANDB_TEAM')}/{env('WANDB_PROJECT')}")
        return weave.op(judge, name="judge_clip"), "on"
    except Exception as e:
        return judge, f"off ({type(e).__name__})"


def find_scenario(vss, scenario, judge=judge_clip, per_camera=3):
    """Search every camera for the scenario and judge each hit.

    Returns {"cameras", "empty_cameras", "hits"}; each hit has camera_id, video, source,
    start_sec, end_sec, description, objects, match and reason.
    """
    cameras = vss.cameras()
    if not cameras:
        raise RuntimeError("no camera_id values found in metadata")

    # Searches run one at a time: parallel searches can push the shared backend past its memory limit.
    per_cam = [vss.search(scenario, cam, per_camera) for cam in cameras]
    raw = [(cam, hit) for cam, results in zip(cameras, per_cam) for hit in results]
    with ThreadPoolExecutor(max_workers=4) as pool:
        verdicts = list(pool.map(
            lambda item: judge(scenario, item[1].get("reasoning_content") or "", detected_objects(item[1])),
            raw,
        ))

    hits = []
    for (cam, hit), verdict in zip(raw, verdicts):
        hits.append({
            "camera_id": hit.get("camera_id") or cam,
            "video": os.path.basename(hit.get("original_video") or hit.get("filename") or "?"),
            "source": hit.get("source"),
            "start_sec": hit.get("segment_start_sec"),
            "end_sec": hit.get("segment_end_sec"),
            "description": " ".join((hit.get("reasoning_content") or "").split()),
            "objects": detected_objects(hit),
            **verdict,
        })
    return {
        "cameras": cameras,
        "empty_cameras": [cam for cam, results in zip(cameras, per_cam) if not results],
        "hits": hits,
    }


def fmt_time(sec):
    if sec is None:
        return "?"
    m, s = divmod(float(sec), 60)
    return f"{int(m):02d}:{s:04.1f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenario", nargs="+", help="scenario in plain words")
    parser.add_argument("--per-camera", type=int, default=3, help="hits kept per camera (default 3)")
    parser.add_argument("--out", default="scenarios.jsonl", help="file for confirmed clips (default scenarios.jsonl)")
    parser.add_argument("--no-weave", action="store_true", help="skip weave tracing")
    args = parser.parse_args()
    scenario = " ".join(args.scenario)

    vss = VSS(env("INGRESS_URL"), env("USERNAME"), env("PASSWORD"))
    judge, weave_status = (judge_clip, "off") if args.no_weave else enable_weave(judge_clip)
    result = find_scenario(vss, scenario, judge, args.per_camera)

    print(f'Scenario: "{scenario}"')
    print(f"Cameras: {len(result['cameras'])}, top {args.per_camera} per camera, judge: {JUDGE_MODEL}, weave: {weave_status}\n")

    current = None
    for hit in result["hits"]:
        if hit["camera_id"] != current:
            print(f"== {hit['camera_id']}")
            current = hit["camera_id"]
        print(f"  [{hit['match'].upper():>6}] {hit['video']}  {fmt_time(hit['start_sec'])} - {fmt_time(hit['end_sec'])}")
        print(textwrap.fill(hit["reason"], width=100, initial_indent="           ", subsequent_indent="           "))
    if result["empty_cameras"]:
        print(f"\nNo hits from: {', '.join(result['empty_cameras'])}")

    fields = ("video", "start_sec", "end_sec", "camera_id", "description", "reason")
    confirmed = [{"scenario": scenario, **{k: h[k] for k in fields}} for h in result["hits"] if h["match"] == "yes"]
    with open(args.out, "w") as f:
        for row in confirmed:
            f.write(json.dumps(row) + "\n")

    print(f"\n{len(result['hits'])} found -> {len(confirmed)} confirmed (saved to {args.out})")


if __name__ == "__main__":
    main()
