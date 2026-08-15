"""Re-runs the 20 curated score>=0.75, name-divergent False Negative cases
through the UPDATED src/mollm_review.py (CRITICAL AUDIT INSTRUCTIONS + 3-step
reasoning) using the real Ollama ensemble, and compares the new per-model
verdicts against the already-known OLD verdicts stored in
mollm_review_decisions (same entities, same retrieval, only the prompt
changed -- a clean before/after on the exact anchoring-bias regime the
diagnostic identified).
"""
import sys, os, json, time, collections
sys.path.insert(0, '/home/ec2-user/clinical_neuro_symbolic_pipeline')
os.chdir('/home/ec2-user/clinical_neuro_symbolic_pipeline')

import duckdb
from src.retrieval import DuckDBHierarchy, GroundingRetriever, GuidelineIndex, VocabularyRetriever
from src.mollm_ensemble import load_validation_records
from src.mollm_review import review_record, build_clients

with open('reports/contradiction_detection/anchoring_bias_targets.json') as fh:
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
            artifact = review_record(rec, retriever, conn, clients=clients)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue
        new_verdicts = [m.get('assessment') for m in artifact.get('models', [])]
        new_reasonings = [m.get('reasoning') for m in artifact.get('models', [])]
        print(f"  OLD verdicts: {t['old_verdicts']}")
        print(f"  NEW verdicts: {new_verdicts}")
        results.append({
            'entity_id': entity_id, 'note_id': note_id, 'entity_label': t['entity_label'],
            'original_text': t['original_text'], 'top1_concept_name': t['top1_concept_name'],
            'top1_score': t['top1_score'], 'gold_span': t['gold_span'],
            'old_verdicts': t['old_verdicts'], 'new_verdicts': new_verdicts,
            'new_assessment_ensemble': artifact.get('assessment'),
            'new_reasonings': new_reasonings,
        })

conn.close()

os.makedirs('reports/contradiction_detection', exist_ok=True)
with open('reports/contradiction_detection/prompt_ab_results.json', 'w') as fh:
    json.dump(results, fh, indent=2, default=str)

# --- summary ---
def classify(v):
    if v is None: return 'ABSTAIN'
    v = str(v).strip().upper()
    if v == 'CORRECT': return 'CONFIRM'
    if v in ('ENTITY_LABEL_INCORRECT','CONCEPT_MAPPING_INCORRECT','BOTH_INCORRECT'): return 'FLAG'
    if v == 'UNCERTAIN': return 'ABSTAIN'
    return 'ABSTAIN'

print("\n" + "="*70)
print("SUMMARY: per-model verdict shift (old CONFIRM -> new ?)")
print("="*70)
flip_counts = collections.Counter()
for r in results:
    old_classes = [classify(v) for v in r['old_verdicts']]
    new_classes = [classify(v) for v in r['new_verdicts']]
    n_flagged_new = sum(1 for c in new_classes if c == 'FLAG')
    n_flagged_old = sum(1 for c in old_classes if c == 'FLAG')
    outcome = f"{n_flagged_old}/{len(old_classes)} old-FLAG -> {n_flagged_new}/{len(new_classes)} new-FLAG"
    flip_counts[outcome] += 1
    print(f"{r['entity_id']:32s} {r['original_text']!r:25s} {outcome}")

print()
for k, v in flip_counts.most_common():
    print(f"  {v:3d}x  {k}")
