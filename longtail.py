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
from concurrent.futures import ThreadPoolExecutor, as_completed

WANDB_URL = "https://api.inference.wandb.ai/v1/chat/completions"
JUDGE_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B"
JUDGE_MAX_TOKENS = 8192
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

DIAGNOSE_MAX_TOKENS = 8192
MAX_INGEST_PROMPT = 800
DIAGNOSE_SYSTEM = (
    "A video archive was searched for a scenario. Every clip in it was described at ingest by "
    "a vision-language model (Cosmos Reason) using a description prompt, and search matches "
    "against those text descriptions. You get the scenario and the reasons a verifier gave for "
    "rejecting the clips search returned. Work out what the descriptions failed to capture "
    "that made search return the wrong clips or miss the right ones. Then write a better "
    "description prompt for the vision-language model that, applied to every clip at ingest, "
    "would make this scenario findable: say exactly what to describe and how to phrase it "
    "(for example motion, distances, who is on foot or inside a vehicle). Keep it general "
    "enough for all footage and under 800 characters. Respond with JSON only, no other text: "
    '{"why_search_missed": "<two short sentences>", "suggested_ingest_prompt": "<prompt under 800 characters>"}'
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

    def search(self, scenario, camera_id, top_k, attempts=5):
        body = {"query": scenario, "top_k": top_k, "llm_top_n": 1, "metadata_filters": {"camera_id": camera_id}}
        for attempt in range(1, attempts + 1):
            try:
                results = request_json(f"{self.backend}/api/v1/search", body, headers=self.auth).get("results", [])
                return results[:top_k]
            except RuntimeError as e:
                gateway = re.search(r"HTTP 50[234]", str(e))
                model_down = re.search(r"not ready|temporarily unavailable", str(e), re.IGNORECASE)
                if attempt == attempts or not (gateway or model_down):
                    raise
                time.sleep(10 * attempt if gateway else 3 * attempt)


def detected_objects(hit):
    try:
        counts = json.loads(hit.get("object_counts") or "{}")
    except json.JSONDecodeError:
        counts = {}
    if counts:
        return ", ".join(f"{label} x{n}" for label, n in sorted(counts.items(), key=lambda kv: -kv[1]))
    return hit.get("object_classes") or "none"


def extract_json(content):
    match = re.search(r"\{.*\}", content or "", re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def chat(system, user, max_tokens):
    headers = {
        "Authorization": f"Bearer {env('WANDB_API_KEY')}",
        "OpenAI-Project": f"{env('WANDB_TEAM')}/{env('WANDB_PROJECT')}",
    }
    body = {
        "model": JUDGE_MODEL,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    resp = request_json(WANDB_URL, body, headers=headers, timeout=180)
    return resp["choices"][0]["message"].get("content")


def judge_clip(scenario: str, description: str, objects: str) -> dict:
    prompt = f"Scenario: {scenario}\n\nClip description: {description}\n\nDetected objects: {objects}"
    verdict = None
    for _ in range(2):
        try:
            verdict = extract_json(chat(JUDGE_SYSTEM, prompt, JUDGE_MAX_TOKENS))
        except Exception as e:
            return {"match": "unsure", "reason": f"Model call failed: {str(e)[:120]}"}
        if verdict is not None:
            break
    if verdict is None:
        return {"match": "unsure", "reason": "Model returned no valid JSON verdict."}
    label = str(verdict.get("match", "")).strip().lower()
    return {
        "match": label if label in VERDICTS else "unsure",
        "reason": str(verdict.get("reason", "")).strip() or "No reason given.",
    }


def trim(text, limit):
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind(".\n"))
    return cut[: end + 1] if end > limit // 2 else cut[: cut.rfind(" ")]


def diagnose_misses(scenario: str, rejections: list) -> dict:
    """Explain why search returned non-matching clips and suggest a better ingest prompt."""
    listing = "\n".join(f"- [{r['match']}] {r['reason']}" for r in rejections)
    try:
        content = chat(DIAGNOSE_SYSTEM, f"Scenario: {scenario}\n\nRejected clips and the verifier's reasons:\n{listing}", DIAGNOSE_MAX_TOKENS)
    except Exception as e:
        return {"why_search_missed": f"Analysis failed: {str(e)[:120]}", "suggested_ingest_prompt": ""}
    data = extract_json(content) or {}
    return {
        "why_search_missed": str(data.get("why_search_missed", "")).strip() or "The model returned no usable analysis.",
        "suggested_ingest_prompt": trim(str(data.get("suggested_ingest_prompt", "")).strip(), MAX_INGEST_PROMPT),
    }


def enable_weave(*fns):
    """Wrap each function in a weave op. Returns (*wrapped, status)."""
    os.environ.setdefault("WEAVE_PRINT_CALL_LINK", "false")
    try:
        import weave

        weave.init(f"{env('WANDB_TEAM')}/{env('WANDB_PROJECT')}")
        return (*(weave.op(fn, name=fn.__name__) for fn in fns), "on")
    except Exception as e:
        return (*fns, f"off ({type(e).__name__})")


def find_scenario(vss, scenario, judge=judge_clip, diagnose=diagnose_misses, per_camera=3, on_event=None):
    """Search every camera for the scenario, judge each hit, then analyze the misses.

    Returns {"cameras", "empty_cameras", "hits", "insight"}; each hit has camera_id, video,
    source, start_sec, end_sec, description, objects, match and reason. insight is
    {"why_search_missed", "suggested_ingest_prompt"}, or None when every hit was confirmed.

    on_event, if given, is called from this thread as the run progresses with:
      {"kind": "camera", "text"}                after each camera search
      {"kind": "searched", "hits"}              once, with the hit list (verdicts still None)
      {"kind": "verdict", "clip", "match", "text"}  as each verdict arrives; clip indexes hits
      {"kind": "info", "text"}                  before the miss analysis
    """
    emit = on_event or (lambda event: None)
    cameras = vss.cameras()
    if not cameras:
        raise RuntimeError("no camera_id values found in metadata")

    # Searches run one at a time: parallel searches can push the shared backend past its memory limit.
    per_cam = []
    for i, cam in enumerate(cameras, 1):
        results = vss.search(scenario, cam, per_camera)
        per_cam.append(results)
        emit({"kind": "camera", "text": f"camera {i}/{len(cameras)} · {cam}: {len(results)} hit{'' if len(results) == 1 else 's'}"})

    hits = [
        {
            "camera_id": hit.get("camera_id") or cam,
            "video": os.path.basename(hit.get("original_video") or hit.get("filename") or "?"),
            "source": hit.get("source"),
            "start_sec": hit.get("segment_start_sec"),
            "end_sec": hit.get("segment_end_sec"),
            "description": " ".join((hit.get("reasoning_content") or "").split()),
            "objects": detected_objects(hit),
            "match": None,
            "reason": None,
        }
        for cam, results in zip(cameras, per_cam)
        for hit in results
    ]
    emit({"kind": "searched", "hits": hits})

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(judge, scenario, h["description"], h["objects"]): i for i, h in enumerate(hits)}
        for done, future in enumerate(as_completed(futures), 1):
            i = futures[future]
            hits[i].update(future.result())
            emit({
                "kind": "verdict",
                "clip": i,
                "match": hits[i]["match"],
                "text": f"clip {done}/{len(hits)} · {hits[i]['camera_id']}: {hits[i]['reason']}",
            })

    rejections = [h for h in hits if h["match"] != "yes"]
    if rejections:
        emit({"kind": "info", "text": f"Analyzing why search missed ({len(rejections)} rejected clips)..."})
    return {
        "cameras": cameras,
        "empty_cameras": [cam for cam, results in zip(cameras, per_cam) if not results],
        "hits": hits,
        "insight": diagnose(scenario, rejections) if rejections else None,
    }


def fmt_time(sec):
    if sec is None:
        return "?"
    m, s = divmod(float(sec), 60)
    return f"{int(m):02d}:{s:04.1f}"


def print_event(event):
    if event["kind"] == "searched":
        return
    mark = {"camera": "•", "info": "…"}.get(event["kind"]) or ("✓" if event["match"] == "yes" else "✗")
    print(f"{mark} {event['text']}", file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenario", nargs="+", help="scenario in plain words")
    parser.add_argument("--per-camera", type=int, default=3, help="hits kept per camera (default 3)")
    parser.add_argument("--out", default="scenarios.jsonl", help="file for confirmed clips (default scenarios.jsonl)")
    parser.add_argument("--no-weave", action="store_true", help="skip weave tracing")
    args = parser.parse_args()
    scenario = " ".join(args.scenario)

    vss = VSS(env("INGRESS_URL"), env("USERNAME"), env("PASSWORD"))
    if args.no_weave:
        judge, diagnose, weave_status = judge_clip, diagnose_misses, "off"
    else:
        judge, diagnose, weave_status = enable_weave(judge_clip, diagnose_misses)
    result = find_scenario(vss, scenario, judge, diagnose, args.per_camera, on_event=print_event)

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

    insight = result["insight"]
    if insight:
        wrap = lambda text: textwrap.fill(text, width=100, initial_indent="  ", subsequent_indent="  ")
        print("\nWhat your archive can't see")
        print(wrap(insight["why_search_missed"]))
        print(f"\nSuggested ingest prompt ({len(insight['suggested_ingest_prompt'])} characters):")
        print(wrap(insight["suggested_ingest_prompt"]))


if __name__ == "__main__":
    main()
