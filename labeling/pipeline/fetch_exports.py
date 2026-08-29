"""Download Neuronpedia bulk activations + features exports for
gemma-3-27b / 40-gemmascope-2-res-16k into sibling dirs of explanations/.

Resumable: skips files that already exist with nonzero size.
"""

import concurrent.futures as cf
import pathlib
import re
import sys
import urllib.request

import os

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "gemma-3-27b/40-gemmascope-2-res-16k"
BUCKET = "https://neuronpedia-datasets.s3.amazonaws.com"
# OUT_ROOT overrides the destination (e.g. a non-iCloud-synced cache dir)
ROOT = pathlib.Path(os.environ.get("OUT_ROOT",
                    pathlib.Path(__file__).resolve().parent.parent))
DIRNAME = SOURCE.replace("/", "-")


def list_keys(prefix):
    keys, token = [], None
    while True:
        url = f"{BUCKET}/?list-type=2&prefix={prefix}"
        if token:
            url += f"&continuation-token={urllib.parse.quote(token)}"
        xml = urllib.request.urlopen(url, timeout=60).read().decode()
        keys += re.findall(r"<Key>([^<]+)</Key>", xml)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if not m:
            return keys
        token = m.group(1)


def fetch(key, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return f"skip {dest.name}"
    tmp = dest.with_suffix(".part")
    for attempt in range(6):
        try:
            with urllib.request.urlopen(f"{BUCKET}/{key}", timeout=120) as r, open(tmp, "wb") as out:
                while chunk := r.read(1 << 20):
                    out.write(chunk)
            tmp.rename(dest)
            return f"got  {dest.name} ({dest.stat().st_size // 1024} KB)"
        except Exception as e:
            if attempt == 5:
                return f"FAILED {dest.name}: {e}"
            import time
            time.sleep(min(90, 15 * (attempt + 1)))


def main():
    jobs = []
    for kind in ("activations", "features", "explanations"):
        outdir = ROOT / kind / DIRNAME
        outdir.mkdir(parents=True, exist_ok=True)
        keys = [k for k in list_keys(f"v1/{SOURCE}/{kind}/") if k.endswith(".jsonl.gz")]
        print(f"{kind}: {len(keys)} files")
        jobs += [(k, outdir / k.rsplit("/", 1)[1]) for k in keys]
    with cf.ThreadPoolExecutor(4) as ex:
        for msg in ex.map(lambda j: fetch(*j), jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
