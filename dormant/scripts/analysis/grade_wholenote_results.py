"""Grades src/mollm_wholenote_ensemble.py's output against the exact same
gold-crosswalk confusion-matrix methodology used throughout
docs/2026-08-15_Contradiction_Detection_Analysis.md, so the whole-note
experiment's numbers are directly comparable to the per-entity numbers
already measured (contradiction_matrix.py, fn_breakdown.py, prompt_ab_test*.py).

Also cross-references the known anchoring-bias trap entities from the earlier
curated A/B tests (same entity_ids) to show, side by side: gold truth, the
OLD per-entity verdict, and this whole-note verdict, for the exact same
entities.
"""
import sys, os, collections, json, re, glob
sys.path.insert(0, '/home/ec2-user/clinical_neuro_symbolic_pipeline')
import duckdb
from scripts.score_gold_recall import load_gold, overlaps, GOLD_CANDIDATES
from evaluation.cal_eval import _first_existing
from src.retrieval import VocabularyRetriever

RESULTS_DIR = 'reports/contradiction_detection/wholenote_results'

def classify_verdict(v):
    if v is None: return 'ABSTAINS'
    v = str(v).strip().upper()
    if v == 'CORRECT': return 'CONFIRMS'
    if v in ('ENTITY_LABEL_INCORRECT', 'CONCEPT_MAPPING_INCORRECT', 'BOTH_INCORRECT'): return 'FLAGS'
    if v == 'UNCERTAIN': return 'ABSTAINS'
    return 'ABSTAINS'

def ensemble_judgment(verdicts):
    classes = [classify_verdict(v) for v in verdicts]
    votes = collections.Counter(c for c in classes if c != 'ABSTAINS')
    if not votes: return 'NO_CONSENSUS'
    top, top_n = votes.most_common(1)[0]
    return top if top_n > sum(votes.values()) - top_n else 'NO_CONSENSUS'

result_files = sorted(glob.glob(os.path.join(RESULTS_DIR, '*.json')))
if not result_files:
    print(f'no result files found in {RESULTS_DIR} yet.')
    sys.exit(1)

note_ids = [os.path.basename(f)[:-5] for f in result_files]
print(f'grading {len(note_ids)} note(s): {note_ids}')

conn = duckdb.connect('db/kg2_lexical_store.duckdb', read_only=True)
vocab = VocabularyRetriever(conn)
gold_path = _first_existing(GOLD_CANDIDATES, 'gold')
gold_rows = load_gold(gold_path, note_ids)
gold_by_note = collections.defaultdict(list)
for g in gold_rows:
    gold_by_note[g['note_id']].append(g)

placeholders = ','.join('?' * len(note_ids))
span_by_entity = {r[0]: (r[1], r[2], r[3]) for r in conn.execute(f'''
    SELECT entity_id, note_id, orig_start, orig_end FROM extracted_entities
    WHERE note_id IN ({placeholders})
''', note_ids).fetchall()}

matrix = collections.Counter()
per_entity = {}
issues_total = 0

for f in result_files:
    with open(f) as fh:
        data = json.load(fh)
    note_id = data['note_id']
    issues_total += len(data.get('issues', []))
    for entity_id, ent in data['entities'].items():
        candidates = ent.get('candidates') or []
        if not candidates:
            continue
        span = span_by_entity.get(entity_id)
        if not span:
            continue
        _, orig_start, orig_end = span
        gold = gold_by_note.get(note_id, [])
        overlapping = [g for g in gold if overlaps(orig_start, orig_end, g['start'], g['end'])]
        if not overlapping:
            continue
        gold_codes = {g['concept_id'] for g in overlapping}
        top1_code = vocab.snomed_code_for_concept(candidates[0].get('omop_concept_id'))
        ground_truth = 'CORRECT' if (top1_code and top1_code in gold_codes) else 'WRONG'

        verdicts = [m.get('assessment') for m in ent.get('models', [])]
        judgment = ensemble_judgment(verdicts)
        matrix[(ground_truth, judgment)] += 1
        per_entity[entity_id] = {
            'note_id': note_id, 'original_text': ent.get('original_text'),
            'top1_concept_name': candidates[0].get('concept_name'),
            'gold_span': overlapping[0].get('span'), 'ground_truth': ground_truth,
            'verdicts': verdicts, 'judgment': judgment,
        }

conn.close()

print(f'\ntotal issues logged across all chunks/models: {issues_total}')
print(f'total classified: {sum(matrix.values())}')
for gt in ('WRONG', 'CORRECT'):
    for j in ('CONFIRMS', 'FLAGS', 'NO_CONSENSUS'):
        print(f'  gold_top1={gt:7s} ensemble={j:12s} n={matrix[(gt,j)]}')

tp = matrix[('WRONG', 'FLAGS')]
fn = matrix[('WRONG', 'CONFIRMS')]
tn = matrix[('CORRECT', 'CONFIRMS')]
fp = matrix[('CORRECT', 'FLAGS')]
print()
print(f'TP (caught a real error):        {tp}')
print(f'FN (rubber-stamped a real error): {fn}')
print(f'TN (validated a good match):      {tn}')
print(f'FP (wrongly overturned a good match): {fp}')
if tp + fn:
    print(f'contradiction-detection RECALL: {100*tp/(tp+fn):.1f}%')
if tn + fp:
    print(f'validation SPECIFICITY: {100*tn/(tn+fp):.1f}%')
if tp + fp:
    print(f'FLAG precision: {100*tp/(tp+fp):.1f}%')

os.makedirs('reports/contradiction_detection', exist_ok=True)
with open('reports/contradiction_detection/wholenote_matrix.json', 'w') as fh:
    json.dump({'matrix': {str(k): v for k, v in matrix.items()}, 'per_entity': per_entity},
               fh, indent=2, default=str)

# --- cross-reference against the known anchoring-bias trap entities ---
print(f'\n{"="*70}\nCross-reference: known trap entities (from prior per-entity A/B tests)\n{"="*70}')
for targets_file, old_label in [
    ('reports/contradiction_detection/anchoring_bias_targets.json', 'obj3 (per-entity)'),
    ('reports/contradiction_detection/anchoring_bias_targets_obj2.json', 'obj2 (per-entity)'),
]:
    if not os.path.exists(targets_file):
        continue
    with open(targets_file) as fh:
        targets = json.load(fh)
    for t in targets:
        eid = t['entity_id']
        if eid not in per_entity:
            continue
        pe = per_entity[eid]
        print(f"{t['original_text']!r:20s} ({old_label:20s}) old_verdicts={t['old_verdicts']} "
              f"-> wholenote_verdicts={pe['verdicts']} wholenote_judgment={pe['judgment']} "
              f"gold={pe['ground_truth']}")
