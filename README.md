# LongTail

LongTail finds the rare, risky moments in hours of unwatched driving, street and warehouse footage, double-checks each one with an AI, and saves them as a test set for self-driving and robot teams.

## How it works

1. **VAST search per camera.** LongTail reads every `camera_id` from the VSS metadata and runs a semantic search for your scenario on each camera separately, keeping the top 3 clips per camera.
2. **W&B Nemotron check.** Each clip's description and detected objects go to `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B` on W&B Inference, which answers in JSON: `yes`, `no` or `unsure`, plus a one-sentence reason. When a scenario involves a person and a vehicle, the person must be on foot; riding or driving doesn't count. After judging, one more Nemotron call reads all the `no` and `unsure` reasons and explains why search missed, with a better Cosmos ingest prompt (under 800 characters) that would make the scenario findable.
3. **Confirmed clips saved.** Clips judged `yes` are written to `scenarios.jsonl` with the video name, start and end time, camera, description and reason.
4. **Web page at `/app/longtail`.** A FastAPI app runs the same pipeline from a text box and plays each confirmed clip, shows the miss analysis in a "What your archive can't see" box, and lists the rejected clips in a collapsed list.

## What is real

- Semantic search over the team's VAST video archive, one camera at a time.
- The AI check: every clip is judged by Nemotron through W&B Inference.
- The miss analysis: a real Nemotron call over that run's rejection reasons, producing a suggested ingest prompt.
- Video playback: clips stream through the app's server, so no token reaches the browser.
- Weave traces of every AI check when run from the command line.

## What is limited

- Verdicts come from each clip's text description and detected objects, not from the video itself.
- Weave tracing is off in the deployed web app.
- A run takes one to three minutes; searches run one at a time so the shared backend doesn't run out of memory.
- Verdicts on borderline clips can change between runs, even at temperature 0.
- The suggested ingest prompt isn't applied automatically; using it means re-ingesting the footage with it.
- Results depend on how the clips were described at ingest. If a description doesn't mention something, LongTail can't find or confirm it.

## How to run it

These need `INGRESS_URL`, `USERNAME`, `PASSWORD`, `WANDB_API_KEY`, `WANDB_TEAM` and `WANDB_PROJECT` in the environment (preset on the Builders Challenge VM).

**Command line**

```bash
python3 longtail.py "person close to a moving vehicle"
python3 longtail.py --per-camera 5 --out found.jsonl forklift near a pedestrian
```

With Weave tracing, use a virtual environment that has `weave` installed. The Builders Challenge VM's Python has no bundled pip, so bootstrap it:

```bash
python3 -m venv --without-pip .venv
curl -sSL https://bootstrap.pypa.io/get-pip.py | .venv/bin/python
.venv/bin/pip install weave
.venv/bin/python longtail.py "person close to a moving vehicle"
```

**Web app**

Deploy to the team cluster at `/app/longtail`:

```bash
./deploy.sh web longtail.py
```

Then open http://team-1-app.thecosmoslabs.com/app/longtail, type a scenario and click Run.
