"""Objective 2 analogue of prompt_ab_test.py -- re-runs the 20 curated
name-divergent, high-confidence False Negatives from mollm_decisions through
the UPDATED src/mollm_ensemble.py SYSTEM_PROMPT (harmonized PROVENANCE
wording + DEVIL'S ADVOCATE + CONCEPTUAL FIREWALL, added 2026-08-15 to close
the recall gap vs. Objective 3 -- see the dated comment above SYSTEM_PROMPT
in that file) using the real Ollama ensemble, and compares against the
already-known OLD verdicts stored in mollm_decisions.
"""
import sys, os, json, time, collections
sys.path.insert(0, '/home/ec2-user/clinical_neuro_symbolic_pipeline')
os.chdir('/home/ec2-user/clinical_neuro_symbolic_pipeline')

import duckdb
from src.retrieval import DuckDBHierarchy, GroundingRetriever, GuidelineIndex, VocabularyRetriever
from src.mollm_ensemble import load_validation_records, validate_record
from src.llm_client import build_clients

with open('reports/contradiction_detection/anchoring_bias_targets_obj2.json') as fh:
    targets = json.load(fh)

by_entity = {t['entity_id']: t for t in targets}
notes_needed = sorted({t['note_id'] for t in targets})

TRIPLETS_CANDIDATES = [
    "data/local_triplets_db2_v6_cleaned_grounded_rules_added",
    "data/local_triplets_db2_v6_cleaned_grounded",
    "data/local_triplets_db2_v6_cleaned",
]
triplets = next((p for p in TRIPLETS_CANDIDATES if os.path.exists(p)), None)
print(f"guideline corpus: {triplets}")

conn = duckdb.connect('db/kg2_lexical_store.duckdb', read_only=True)
index = GuidelineIndex(triplets)
vocab = VocabularyRetriever(conn)
retriever = GroundingRetriever(index, vocab, hierarchy=DuckDBHierarchy(conn))
clients = build_clients()
print(f"guideline KG: {index.stats['nodes']} nodes, {index.stats['rules']} rules")
print(f"{len(targets)} target entities across {len(notes_needed)} notes\n")

results = []
start = time.time()
done = 0
for note_id in notes_needed:
    records = load_validation_records(conn, note_id)
    rec_by_id = {r['entity_id']: r for r in records}
    for entity_id, t in by_entity.items():
        if t['note_id'] != note_id:
            continue
        rec = rec_by_id.get(entity_id)
        if rec is None:
            print(f"SKIP {entity_id}: not found in current load_validation_records() for {note_id}")
            continue
        done += 1
        elapsed = time.time() - start
        print(f"[{done}/{len(targets)}, {elapsed/60:.1f}m] {entity_id} {t['original_text']!r} -> {t['top1_concept_name']!r} (score={t['top1_score']})")
        try:
            artifact = validate_record(rec, retriever, clients=clients)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue
        new_verdicts = [m.get('verdict') for m in artifact.get('models', [])]
        new_reasonings = [m.get('reasoning') for m in artifact.get('models', [])]
        print(f"  OLD verdicts: {t['old_verdicts']}")
        print(f"  NEW verdicts: {new_verdicts}")
        results.append({
            'entity_id': entity_id, 'note_id': note_id, 'entity_label': t['entity_label'],
            'original_text': t['original_text'], 'top1_concept_name': t['top1_concept_name'],
            'top1_score': t['top1_score'], 'gold_span': t['gold_span'],
            'old_verdicts': t['old_verdicts'], 'new_verdicts': new_verdicts,
            'new_mode': artifact.get('mode'),
            'new_reasonings': new_reasonings,
        })

conn.close()

os.makedirs('reports/contradiction_detection', exist_ok=True)
with open('reports/contradiction_detection/prompt_ab_results_obj2.json', 'w') as fh:
    json.dump(results, fh, indent=2, default=str)

# --- summary (unified classify_verdict, works across all three modes) ---
import re
_RESOLVED_RE = re.compile(r'^RESOLVED_TO_CANDIDATE_(\d+)$')
def classify(v):
    if v is None: return 'ABSTAIN'
    v = str(v).strip().upper()
    if v in ('CORRECT', 'SUPPORTED'): return 'CONFIRM'
    if v == 'RESOLVED_TO_CANDIDATE_1': return 'CONFIRM'
    if v in ('CONTRADICTED', 'NONE_CORRECT', 'ENTITY_LABEL_INCORRECT',
              'CONCEPT_MAPPING_INCORRECT', 'BOTH_INCORRECT'): return 'FLAG'
    m = _RESOLVED_RE.match(v)
    if m and int(m.group(1)) > 1: return 'FLAG'
    return 'ABSTAIN'

def ensemble(verdicts):
    classes = [classify(v) for v in verdicts]
    votes = collections.Counter(c for c in classes if c != 'ABSTAIN')
    if not votes: return 'NO_CONSENSUS'
    top, n = votes.most_common(1)[0]
    return top if n > sum(votes.values()) - n else 'NO_CONSENSUS'

print("\n" + "="*70)
print("SUMMARY: per-model verdict shift (old CONFIRM -> new ?)")
print("="*70)
flips = 0
old_flag_total = new_flag_total = 0
for r in results:
    oc = ensemble(r['old_verdicts']); nc = ensemble(r['new_verdicts'])
    old_flag_total += sum(1 for v in r['old_verdicts'] if classify(v) == 'FLAG')
    new_flag_total += sum(1 for v in r['new_verdicts'] if classify(v) == 'FLAG')
    print(f"{r['entity_id']:32s} {r['original_text']!r:20s} old_ensemble={oc:12s} new_ensemble={nc:12s} {'<-- FLIP' if oc=='CONFIRM' and nc=='FLAG' else ''}")
    if oc == 'CONFIRM' and nc == 'FLAG':
        flips += 1

print()
print(f'ensemble-level CONFIRM->FLAG flips: {flips}/{len(results)}')
if results:
    print(f'per-model FLAG count: old {old_flag_total}/{3*len(results)} -> new {new_flag_total}/{3*len(results)}')
