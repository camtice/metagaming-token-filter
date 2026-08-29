"""Submit relabel/requests.jsonl to the Anthropic Message Batches API, poll
until done, and save raw results. No SDK dependency — plain urllib.

Usage:
  ANTHROPIC_API_KEY=... python3 relabel/run_batch.py [--limit 100]
  ANTHROPIC_API_KEY=... python3 relabel/run_batch.py --poll <batch_id>
"""

import argparse
import json
import os
import pathlib
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
API = "https://api.anthropic.com/v1/messages/batches"


def call(url, payload=None):
    req = urllib.request.Request(url, headers={
        "x-api-key": os.environ["ANTHROPIC_API_KEY"],
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }, data=json.dumps(payload).encode() if payload else None)
    # large uploads on slow links need a generous socket timeout
    return json.load(urllib.request.urlopen(req, timeout=3600 if payload else 300))


def poll(batch_id, tag):
    while True:
        b = call(f"{API}/{batch_id}")
        c = b["request_counts"]
        print(f"{batch_id}: {b['processing_status']} "
              f"(ok={c['succeeded']} err={c['errored']} pending={c['processing']})",
              flush=True)
        if b["processing_status"] == "ended":
            break
        time.sleep(60)
    out = ROOT / "relabel" / f"batch_results_{tag}.jsonl"
    req = urllib.request.Request(b["results_url"], headers={
        "x-api-key": os.environ["ANTHROPIC_API_KEY"],
        "anthropic-version": "2023-06-01",
    })
    out.write_bytes(urllib.request.urlopen(req, timeout=600).read())
    print(f"results -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", default="requests.jsonl",
                    help="requests file inside relabel/ (default requests.jsonl)")
    ap.add_argument("--tag", help="override the results-file tag")
    ap.add_argument("--limit", type=int, help="submit only the first N requests")
    ap.add_argument("--skip", type=int, help="skip the first N requests")
    ap.add_argument("--calibration", action="store_true",
                    help="submit only the previously hand-verified latents")
    ap.add_argument("--model", help="override the model id in every request")
    ap.add_argument("--chunk", type=int, default=6000,
                    help="requests per batch (smaller = flaky-network friendly)")
    ap.add_argument("--poll", help="skip submission; poll this batch id")
    args = ap.parse_args()

    if args.poll:
        poll(args.poll, args.poll[-8:])
        return

    reqs = [json.loads(l) for l in open(ROOT / "relabel" / args.requests)]
    if args.calibration:
        verdicts = json.load(open(ROOT / "results" / "gemma3_L40_16k_verdicts.json"))
        wanted = {f"latent-{x['index']}" for x in verdicts}
        reqs = [r for r in reqs if r["custom_id"] in wanted]
    if args.skip:
        reqs = reqs[args.skip:]
    if args.limit:
        reqs = reqs[:args.limit]
    if args.model:
        mythos = "fable" in args.model or "mythos" in args.model
        for r in reqs:
            r["params"]["model"] = args.model
            # temperature is deprecated on the Claude 5 family, and thinking
            # defaults ON there — disable it and leave headroom for the answer.
            # Mythos-class models reject thinking.disabled (adaptive only), so
            # leave thinking unset and give the answer room after any thinking.
            r["params"].pop("temperature", None)
            if mythos:
                r["params"].pop("thinking", None)
                r["params"]["max_tokens"] = 1500
            else:
                r["params"]["thinking"] = {"type": "disabled"}
                r["params"]["max_tokens"] = 300
    # stay under the 256 MB / 100k-request per-batch limits
    ids = []
    for n, start in enumerate(range(0, len(reqs), args.chunk)):
        chunk = reqs[start:start + args.chunk]
        for attempt in range(10):
            try:
                b = call(API, {"requests": chunk})
                break
            except Exception as e:
                print(f"chunk {n} attempt {attempt + 1} failed: {e}", flush=True)
                if attempt == 9:
                    raise
                time.sleep(min(120, 20 * (attempt + 1)))
        ids.append(b["id"])
        print(f"submitted chunk {n} ({len(chunk)} requests) as {b['id']}", flush=True)
    tag = ("calibration" if args.calibration else
           f"limit{args.limit}" if args.limit else "full")
    if args.model:
        tag += "_" + args.model.replace("claude-", "").replace("-", "")
    if args.tag:
        tag = args.tag
    (ROOT / "relabel" / f"batch_id_{tag}.txt").write_text("\n".join(ids))
    for n, bid in enumerate(ids):
        poll(bid, f"{tag}_chunk{n}")


if __name__ == "__main__":
    main()
