"""Contradiction-detection confusion matrix: does the LLM layer correctly
catch Stage 2's errors, or rubber-stamp them?

Ground truth axis: is Stage 2b's own top-1 candidate (candidates[0], BEFORE
any Stage 3 involvement) correct against gold?

LLM axis: per-model verdict, classified into CONFIRMS / FLAGS / ABSTAINS,
unified across both source tables and all three mollm_decisions modes:
  CONFIRMS: assessment=='CORRECT' | verdict=='SUPPORTED' | verdict=='RESOLVED_TO_CANDIDATE_1'
  FLAGS:    assessment in INCORRECT variants | verdict=='CONTRADICTED' | verdict=='NONE_CORRECT'
            | verdict=='RESOLVED_TO_CANDIDATE_N' for N>1
  ABSTAINS: assessment=='UNCERTAIN' | verdict=='INSUFFICIENT_EVIDENCE'
Ensemble judgment = majority vote among CONFIRMS/FLAGS (ties/all-abstain -> NO_CONSENSUS).
"""
import sys, collections, json, re
sys.path.insert(0, '/home/ec2-user/clinical_neuro_symbolic_pipeline')
import duckdb
from scripts.score_gold_recall import load_gold, overlaps, GOLD_CANDIDATES
from evaluation.cal_eval import _first_existing
from src.retrieval import VocabularyRetriever

NOTE_IDS = '10043750-DS-6,10371195-DS-9,10848570-DS-12,10860165-DS-24,10912090-DS-33,11134545-DS-21,11532659-DS-11,11649745-DS-4,11838076-DS-20,11997336-DS-3,12128814-DS-15,12247014-DS-9,12314513-DS-16,12545016-DS-17,12962702-DS-14,12970259-DS-4,13164440-DS-18,14102739-DS-16,14280440-DS-8,14975962-DS-19,15853461-DS-4,16393593-DS-5,16991646-DS-11,17739994-DS-31,18570237-DS-10,14490470-DS-11,17751158-DS-19,19442119-DS-15'.split(',')

_RESOLVED_RE = re.compile(r'^RESOLVED_TO_CANDIDATE_(\d+)$')

def classify_verdict(v):
    if v is None:
        return 'ABSTAINS'
    v = str(v).strip().upper()
    if v in ('CORRECT', 'SUPPORTED'):
        return 'CONFIRMS'
    if v == 'RESOLVED_TO_CANDIDATE_1':
        return 'CONFIRMS'
    if v in ('CONTRADICTED', 'NONE_CORRECT', 'ENTITY_LABEL_INCORRECT',
             'CONCEPT_MAPPING_INCORRECT', 'BOTH_INCORRECT'):
        return 'FLAGS'
    m = _RESOLVED_RE.match(v)
    if m and int(m.group(1)) > 1:
        return 'FLAGS'
    if v in ('INSUFFICIENT_EVIDENCE', 'UNCERTAIN'):
        return 'ABSTAINS'
    return 'ABSTAINS'  # unparseable / unknown -- treat as no signal, not a silent confirm

def ensemble_judgment(verdicts):
    classes = [classify_verdict(v) for v in verdicts]
    votes = collections.Counter(c for c in classes if c != 'ABSTAINS')
    if not votes:
        return 'NO_CONSENSUS'
    top, top_n = votes.most_common(1)[0]
    if top_n > sum(votes.values()) - top_n:  # strict majority among non-abstains
        return top
    return 'NO_CONSENSUS'

conn = duckdb.connect('db/kg2_lexical_store.duckdb', read_only=True)
vocab = VocabularyRetriever(conn)

gold_path = _first_existing(GOLD_CANDIDATES, 'gold')
gold_rows = load_gold(gold_path, NOTE_IDS)
gold_by_note = collections.defaultdict(list)
for g in gold_rows:
    gold_by_note[g['note_id']].append(g)

placeholders = ','.join('?' * len(NOTE_IDS))

matrix = collections.Counter()  # (ground_truth, ensemble_judgment) -> count
examples = collections.defaultdict(list)

def process(rows, models_key):
    for entity_id, note_id, orig_start, orig_end, cands_json, models_json in rows:
        candidates = json.loads(cands_json) if isinstance(cands_json, str) else (cands_json or [])
        if not candidates:
            continue
        gold = gold_by_note.get(note_id, [])
        overlapping = [g for g in gold if overlaps(orig_start, orig_end, g['start'], g['end'])]
        if not overlapping:
            continue
        gold_codes = {g['concept_id'] for g in overlapping}
        top1_code = vocab.snomed_code_for_concept(candidates[0].get('omop_concept_id'))
        ground_truth = 'CORRECT' if (top1_code and top1_code in gold_codes) else 'WRONG'

        models = json.loads(models_json) if isinstance(models_json, str) else (models_json or [])
        verdicts = [m.get(models_key) for m in models]
        judgment = ensemble_judgment(verdicts)
        matrix[(ground_truth, judgment)] += 1
        if len(examples[(ground_truth, judgment)]) < 3:
            examples[(ground_truth, judgment)].append((entity_id, note_id, verdicts))

# --- mollm_decisions (Objective 2) ---
rows_d = conn.execute(f'''
    SELECT e.entity_id, e.note_id, e.orig_start, e.orig_end, n.candidates, d.models
    FROM mollm_decisions d
    JOIN extracted_entities e ON e.entity_id = d.entity_id
    JOIN normalized_entities n ON n.entity_id = d.entity_id
    WHERE d.is_test = TRUE AND d.error IS NULL AND d.note_id IN ({placeholders})
''', NOTE_IDS).fetchall()
process(rows_d, 'verdict')

print('=== mollm_decisions (Objective 2) contradiction-detection matrix ===')
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
    print(f'contradiction-detection RECALL (of wrong top-1s, % caught): {100*tp/(tp+fn):.1f}%')
if tn + fp:
    print(f'validation SPECIFICITY (of correct top-1s, % correctly passed): {100*tn/(tn+fp):.1f}%')
if tp + fp:
    print(f'FLAG precision (of things flagged, % actually wrong): {100*tp/(tp+fp):.1f}%')

import os
os.makedirs('reports/contradiction_detection', exist_ok=True)
with open('reports/contradiction_detection/contradiction_matrix_decisions.json', 'w') as fh:
    json.dump({str(k): v for k, v in matrix.items()}, fh, indent=2, default=str)

conn.close()

# --- mollm_review_decisions (Objective 3) ---
conn2 = duckdb.connect('/home/ec2-user/clinical_neuro_symbolic_pipeline/db/kg2_lexical_store.duckdb', read_only=True)
matrix2 = collections.Counter()
examples2 = collections.defaultdict(list)

rows_r = conn2.execute(f'''
    SELECT e.entity_id, e.note_id, e.orig_start, e.orig_end, n.candidates, r.models
    FROM mollm_review_decisions r
    JOIN extracted_entities e ON e.entity_id = r.entity_id
    JOIN normalized_entities n ON n.entity_id = r.entity_id
    WHERE r.is_test = TRUE AND r.error IS NULL AND r.note_id IN ({placeholders})
''', NOTE_IDS).fetchall()

def process2(rows, models_key, matrix_out, examples_out):
    for entity_id, note_id, orig_start, orig_end, cands_json, models_json in rows:
        candidates = json.loads(cands_json) if isinstance(cands_json, str) else (cands_json or [])
        if not candidates:
            continue
        gold = gold_by_note.get(note_id, [])
        overlapping = [g for g in gold if overlaps(orig_start, orig_end, g['start'], g['end'])]
        if not overlapping:
            continue
        gold_codes = {g['concept_id'] for g in overlapping}
        top1_code = vocab.snomed_code_for_concept(candidates[0].get('omop_concept_id'))
        ground_truth = 'CORRECT' if (top1_code and top1_code in gold_codes) else 'WRONG'
        models = json.loads(models_json) if isinstance(models_json, str) else (models_json or [])
        verdicts = [m.get(models_key) for m in models]
        judgment = ensemble_judgment(verdicts)
        matrix_out[(ground_truth, judgment)] += 1
        if len(examples_out[(ground_truth, judgment)]) < 3:
            examples_out[(ground_truth, judgment)].append((entity_id, note_id, verdicts))

process2(rows_r, 'assessment', matrix2, examples2)

print()
print('=== mollm_review_decisions (Objective 3) contradiction-detection matrix ===')
print(f'total classified: {sum(matrix2.values())}')
for gt in ('WRONG', 'CORRECT'):
    for j in ('CONFIRMS', 'FLAGS', 'NO_CONSENSUS'):
        print(f'  gold_top1={gt:7s} ensemble={j:12s} n={matrix2[(gt,j)]}')

tp2 = matrix2[('WRONG', 'FLAGS')]
fn2 = matrix2[('WRONG', 'CONFIRMS')]
tn2 = matrix2[('CORRECT', 'CONFIRMS')]
fp2 = matrix2[('CORRECT', 'FLAGS')]
print()
print(f'TP (caught a real error):        {tp2}')
print(f'FN (rubber-stamped a real error): {fn2}')
print(f'TN (validated a good match):      {tn2}')
print(f'FP (wrongly overturned a good match): {fp2}')
if tp2 + fn2:
    print(f'contradiction-detection RECALL: {100*tp2/(tp2+fn2):.1f}%')
if tn2 + fp2:
    print(f'validation SPECIFICITY: {100*tn2/(tn2+fp2):.1f}%')
if tp2 + fp2:
    print(f'FLAG precision: {100*tp2/(tp2+fp2):.1f}%')
conn2.close()
