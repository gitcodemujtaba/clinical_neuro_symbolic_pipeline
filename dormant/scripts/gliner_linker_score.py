"""scripts/gliner_linker_score.py -- standalone batch scorer, run ONLY inside
.venv_gliner_linker (Python 3.11 + glinker==0.1.1 + gliner==0.2.25 pinned --
gliner>=0.2.28 breaks GLinkerModel.forward() with a str/tensor type error,
confirmed empirically 2026-08-18).

WHY A SEPARATE PROCESS/VENV, NOT AN IMPORT. glinker requires Python >=3.10
and a transformers==5.x-era checkpoint format, both incompatible with this
project's main Python 3.9 / transformers 4.57 environment (which has its own
hard-won version pins -- scispaCy/medspacy require spacy<3.8, GLiNER-BioMed
is tested against transformers 4.x). Isolating this in its own venv and
talking to it via a subprocess + JSON file is the same pattern already used
for Ollama (a separately-managed model service, not something embedded
in-process) -- see docs discussion 2026-08-18.

Usage (from the MAIN pipeline's Python 3.9 process, via subprocess):
    /path/to/.venv_gliner_linker/bin/python3 scripts/gliner_linker_score.py \
        --input requests.json --output results.json [--model large|base]

Input JSON: {"requests": [{"id": <any>, "text": str, "span": {"start": int,
"end": int}, "candidates": [str, ...]}, ...]}
Output JSON: {"results": [{"id": <same id>, "scores": [float, ...]}, ...]}
`scores[i]` corresponds to `candidates[i]` in the matching request -- one
score per candidate, not a single winner, so the caller decides how to use
it (log-only, rerank, threshold) rather than this script deciding for them.

Model loading is the expensive part (~10-45s) -- done ONCE per process
invocation, then every request in the batch is scored against that one
loaded model. Callers should batch as many requests as they have per
invocation rather than calling this once per entity.
"""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Path to input JSON (see module docstring)")
    ap.add_argument("--output", required=True, help="Path to write output JSON")
    ap.add_argument("--model", default="large", choices=["base", "large", "rerank"],
                    help="Which GLiNER-Linker variant to use (default: large, "
                         "the 'maximum accuracy' variant per its own model card). "
                         "'rerank' is a different architecture (ettin-encoder-68m "
                         "cross-encoder, no separate labels_encoder) -- uses the "
                         "L4 API, not L3.")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    with open(args.input) as f:
        payload = json.load(f)
    requests = payload["requests"]

    is_rerank = args.model == "rerank"
    model_name = f"knowledgator/gliner-linker-{args.model}-v1.0"
    if is_rerank:
        from glinker.l4 import L4Component, L4Config
        config = L4Config(model_name=model_name, device=args.device, threshold=0.0)
        comp = L4Component(config)
    else:
        from glinker.l3 import L3Component, L3Config
        config = L3Config(model_name=model_name, device=args.device, threshold=0.0)
        comp = L3Component(config)
    print(f"[gliner_linker_score] loaded {model_name} on {args.device}, "
          f"scoring {len(requests)} request(s)", file=sys.stderr)

    results = []
    for req in requests:
        candidates = req["candidates"]
        if not candidates:
            results.append({"id": req["id"], "scores": []})
            continue
        span = req["span"]
        if is_rerank:
            entities = comp.predict_entities(
                text=req["text"],
                labels=candidates,
                input_spans=[[{"start": span["start"], "end": span["end"]}]],
            )
        else:
            entities = comp.predict_entities(
                text=req["text"],
                labels=candidates,
                input_spans=[{"start": span["start"], "end": span["end"]}],
                span_label_indices=[list(range(len(candidates)))],
            )
        # entities is unordered w.r.t. candidate index -- build a lookup by
        # label text and re-project onto candidates' own order, since the
        # caller's candidates[i] <-> scores[i] contract must hold regardless
        # of whatever internal order the model returns matches in.
        score_by_label = {e.label: e.score for e in entities}
        scores = [score_by_label.get(c, 0.0) for c in candidates]
        results.append({"id": req["id"], "scores": scores})

    with open(args.output, "w") as f:
        json.dump({"results": results}, f)
    print(f"[gliner_linker_score] wrote {len(results)} result(s) to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
