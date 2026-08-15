"""Breaks down the False Negatives (rubber-stamped Stage 2 errors) from the
contradiction-detection matrix along three axes the user asked for:

1. GLiNER label (entity_label) -- which categories dominate the miss list.
2. Stage 2's own top-1 match score/basis -- are high-confidence exact-text
   matches "bullying" the LLM into agreeing even when wrong?
3. Concrete examples per label, for manual semantic-vs-ontology triage.

Reuses the exact same ground-truth/verdict-classification logic as
contradiction_matrix.py (same NOTE_IDS, same classify_verdict/ensemble_judgment),
just captures extra fields per row instead of only tallying the 2x3 matrix.
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
    return 'ABSTAINS'

def ensemble_judgment(verdicts):
    classes = [classify_verdict(v) for v in verdicts]
    votes = collections.Counter(c for c in classes if c != 'ABSTAINS')
    if not votes:
        return 'NO_CONSENSUS'
    top, top_n = votes.most_common(1)[0]
    if top_n > sum(votes.values()) - top_n:
        return top
    return 'NO_CONSENSUS'

def score_bucket(score):
    if score is None:
        return 'unknown'
    try:
        s = float(score)
    except (TypeError, ValueError):
        return 'unknown'
    if s >= 1.0:
        return '1.0 (exact)'
    if s >= 0.9:
        return '0.9-0.999'
    if s >= 0.8:
        return '0.8-0.899'
    if s >= 0.7:
        return '0.7-0.799'
    return '<0.7'

conn = duckdb.connect('db/kg2_lexical_store.duckdb', read_only=True)
vocab = VocabularyRetriever(conn)

gold_path = _first_existing(GOLD_CANDIDATES, 'gold')
gold_rows = load_gold(gold_path, NOTE_IDS)
gold_by_note = collections.defaultdict(list)
for g in gold_rows:
    gold_by_note[g['note_id']].append(g)

placeholders = ','.join('?' * len(NOTE_IDS))


def process(rows, models_key, source_label):
    """Returns list of dicts, one per classified (gold-overlapping) row."""
    out = []
    for (entity_id, note_id, orig_start, orig_end, entity_label, original_text,
         local_context, cands_json, models_json) in rows:
        candidates = json.loads(cands_json) if isinstance(cands_json, str) else (cands_json or [])
        if not candidates:
            continue
        gold = gold_by_note.get(note_id, [])
        overlapping = [g for g in gold if overlaps(orig_start, orig_end, g['start'], g['end'])]
        if not overlapping:
            continue
        gold_codes = {g['concept_id'] for g in overlapping}
        top1 = candidates[0]
        top1_code = vocab.snomed_code_for_concept(top1.get('omop_concept_id'))
        ground_truth = 'CORRECT' if (top1_code and top1_code in gold_codes) else 'WRONG'

        models = json.loads(models_json) if isinstance(models_json, str) else (models_json or [])
        verdicts = [m.get(models_key) for m in models]
        judgment = ensemble_judgment(verdicts)

        out.append({
            'source': source_label,
            'entity_id': entity_id,
            'note_id': note_id,
            'entity_label': entity_label,
            'original_text': original_text,
            'local_context': local_context,
            'top1_concept_name': top1.get('concept_name'),
            'top1_score': top1.get('similarity_score'),
            'top1_basis': top1.get('match_basis'),
            'top1_tier': top1.get('match_tier'),
            'gold_span': overlapping[0].get('span'),
            'gold_concept_id': list(gold_codes)[0] if gold_codes else None,
            'ground_truth': ground_truth,
            'judgment': judgment,
            'verdicts': verdicts,
            'models': models,
        })
    return out


rows_d = conn.execute(f'''
    SELECT e.entity_id, e.note_id, e.orig_start, e.orig_end, e.entity_label,
           e.original_text, e.local_context, n.candidates, d.models
    FROM mollm_decisions d
    JOIN extracted_entities e ON e.entity_id = d.entity_id
    JOIN normalized_entities n ON n.entity_id = d.entity_id
    WHERE d.is_test = TRUE AND d.error IS NULL AND d.note_id IN ({placeholders})
''', NOTE_IDS).fetchall()
all_d = process(rows_d, 'verdict', 'mollm_decisions')

rows_r = conn.execute(f'''
    SELECT e.entity_id, e.note_id, e.orig_start, e.orig_end, e.entity_label,
           e.original_text, e.local_context, n.candidates, r.models
    FROM mollm_review_decisions r
    JOIN extracted_entities e ON e.entity_id = r.entity_id
    JOIN normalized_entities n ON n.entity_id = r.entity_id
    WHERE r.is_test = TRUE AND r.error IS NULL AND r.note_id IN ({placeholders})
''', NOTE_IDS).fetchall()
all_r = process(rows_r, 'assessment', 'mollm_review_decisions')

conn.close()

all_rows = all_d + all_r


def report(rows, title):
    print(f'\n{"="*70}\n{title}\n{"="*70}')
    wrong = [r for r in rows if r['ground_truth'] == 'WRONG']
    fn = [r for r in rows if r['ground_truth'] == 'WRONG' and r['judgment'] == 'CONFIRMS']
    tp = [r for r in rows if r['ground_truth'] == 'WRONG' and r['judgment'] == 'FLAGS']
    print(f'total gold_top1=WRONG: {len(wrong)}  |  FN (rubber-stamped): {len(fn)}  |  TP (caught): {len(tp)}')

    print('\n--- (1) By GLiNER label: recall among WRONG top-1s ---')
    by_label_wrong = collections.Counter(r['entity_label'] for r in wrong)
    by_label_fn = collections.Counter(r['entity_label'] for r in fn)
    rows_out = []
    for label, n_wrong in by_label_wrong.items():
        n_fn = by_label_fn.get(label, 0)
        n_tp = n_wrong - n_fn
        recall = 100 * n_tp / n_wrong if n_wrong else 0
        rows_out.append((label, n_wrong, n_fn, n_tp, recall))
    rows_out.sort(key=lambda x: -x[1])
    print(f'{"label":30s} {"n_wrong":>8s} {"n_FN":>6s} {"n_TP":>6s} {"recall%":>8s}')
    for label, n_wrong, n_fn, n_tp, recall in rows_out:
        print(f'{label or "(none)":30s} {n_wrong:8d} {n_fn:6d} {n_tp:6d} {recall:7.1f}%')

    print('\n--- (2) By Stage 2 top-1 score bucket: recall among WRONG top-1s ---')
    by_bucket_wrong = collections.Counter(score_bucket(r['top1_score']) for r in wrong)
    by_bucket_fn = collections.Counter(score_bucket(r['top1_score']) for r in fn)
    order = ['1.0 (exact)', '0.9-0.999', '0.8-0.899', '0.7-0.799', '<0.7', 'unknown']
    print(f'{"score bucket":15s} {"n_wrong":>8s} {"n_FN":>6s} {"n_TP":>6s} {"recall%":>8s}')
    for b in order:
        n_wrong = by_bucket_wrong.get(b, 0)
        if n_wrong == 0:
            continue
        n_fn = by_bucket_fn.get(b, 0)
        n_tp = n_wrong - n_fn
        recall = 100 * n_tp / n_wrong if n_wrong else 0
        print(f'{b:15s} {n_wrong:8d} {n_fn:6d} {n_tp:6d} {recall:7.1f}%')

    print('\n--- (2b) By Stage 2 top-1 match_basis: recall among WRONG top-1s ---')
    by_basis_wrong = collections.Counter(r['top1_basis'] for r in wrong)
    by_basis_fn = collections.Counter(r['top1_basis'] for r in fn)
    print(f'{"match_basis":25s} {"n_wrong":>8s} {"n_FN":>6s} {"n_TP":>6s} {"recall%":>8s}')
    for basis, n_wrong in sorted(by_basis_wrong.items(), key=lambda x: -x[1]):
        n_fn = by_basis_fn.get(basis, 0)
        n_tp = n_wrong - n_fn
        recall = 100 * n_tp / n_wrong if n_wrong else 0
        print(f'{str(basis):25s} {n_wrong:8d} {n_fn:6d} {n_tp:6d} {recall:7.1f}%')

    print('\n--- (2c) By Stage 2 top-1 match_tier: recall among WRONG top-1s ---')
    by_tier_wrong = collections.Counter(r['top1_tier'] for r in wrong)
    by_tier_fn = collections.Counter(r['top1_tier'] for r in fn)
    print(f'{"match_tier":15s} {"n_wrong":>8s} {"n_FN":>6s} {"n_TP":>6s} {"recall%":>8s}')
    for tier, n_wrong in sorted(by_tier_wrong.items(), key=lambda x: -x[1]):
        n_fn = by_tier_fn.get(tier, 0)
        n_tp = n_wrong - n_fn
        recall = 100 * n_tp / n_wrong if n_wrong else 0
        print(f'{str(tier):15s} {n_wrong:8d} {n_fn:6d} {n_tp:6d} {recall:7.1f}%')

    return fn


fn_d = report(all_d, 'mollm_decisions (Objective 2) -- False Negative breakdown')
fn_r = report(all_r, 'mollm_review_decisions (Objective 3) -- False Negative breakdown')

# --- combined view across both objectives (user asked for the overall picture) ---
fn_all = report(all_rows, 'COMBINED (Objective 2 + 3) -- False Negative breakdown')

# --- (3) concrete examples for semantic-vs-ontology triage, top labels only ---
print(f'\n{"="*70}\nConcrete FN examples (top 4 labels by FN count, 4 examples each)\n{"="*70}')
by_label = collections.defaultdict(list)
for r in fn_all:
    by_label[r['entity_label']].append(r)
top_labels = sorted(by_label.items(), key=lambda kv: -len(kv[1]))[:4]
for label, examples in top_labels:
    print(f'\n### {label} ({len(examples)} FN total, showing up to 4) ###')
    for r in examples[:4]:
        print(f"  entity: {r['original_text']!r}  (note {r['note_id']}, {r['source']})")
        print(f"    context: {(r['local_context'] or '')[:180]!r}")
        print(f"    Stage2 top1: {r['top1_concept_name']!r} (score={r['top1_score']}, basis={r['top1_basis']}, tier={r['top1_tier']})")
        print(f"    gold expects: {r['gold_span']!r} (concept_id={r['gold_concept_id']})")
        print(f"    LLM verdicts: {r['verdicts']}")
        print()

import os
os.makedirs('reports/contradiction_detection', exist_ok=True)
with open('reports/contradiction_detection/fn_breakdown.json', 'w') as fh:
    json.dump({
        'fn_objective2': fn_d,
        'fn_objective3': fn_r,
    }, fh, indent=2, default=str)
