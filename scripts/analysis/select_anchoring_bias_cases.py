import sys, collections, json, re
sys.path.insert(0, '/home/ec2-user/clinical_neuro_symbolic_pipeline')
import duckdb
from scripts.score_gold_recall import load_gold, overlaps, GOLD_CANDIDATES
from evaluation.cal_eval import _first_existing
from src.retrieval import VocabularyRetriever

NOTE_IDS = '10043750-DS-6,10371195-DS-9,10848570-DS-12,10860165-DS-24,10912090-DS-33,11134545-DS-21,11532659-DS-11,11649745-DS-4,11838076-DS-20,11997336-DS-3,12128814-DS-15,12247014-DS-9,12314513-DS-16,12545016-DS-17,12962702-DS-14,12970259-DS-4,13164440-DS-18,14102739-DS-16,14280440-DS-8,14975962-DS-19,15853461-DS-4,16393593-DS-5,16991646-DS-11,17739994-DS-31,18570237-DS-10,14490470-DS-11,17751158-DS-19,19442119-DS-15'.split(',')

_RESOLVED_RE = re.compile(r'^RESOLVED_TO_CANDIDATE_(\d+)$')
def classify_verdict(v):
    if v is None: return 'ABSTAINS'
    v = str(v).strip().upper()
    if v in ('CORRECT','SUPPORTED'): return 'CONFIRMS'
    if v == 'RESOLVED_TO_CANDIDATE_1': return 'CONFIRMS'
    if v in ('CONTRADICTED','NONE_CORRECT','ENTITY_LABEL_INCORRECT','CONCEPT_MAPPING_INCORRECT','BOTH_INCORRECT'): return 'FLAGS'
    m = _RESOLVED_RE.match(v)
    if m and int(m.group(1)) > 1: return 'FLAGS'
    if v in ('INSUFFICIENT_EVIDENCE','UNCERTAIN'): return 'ABSTAINS'
    return 'ABSTAINS'

def ensemble_judgment(verdicts):
    classes = [classify_verdict(v) for v in verdicts]
    votes = collections.Counter(c for c in classes if c != 'ABSTAINS')
    if not votes: return 'NO_CONSENSUS'
    top, top_n = votes.most_common(1)[0]
    if top_n > sum(votes.values()) - top_n: return top
    return 'NO_CONSENSUS'

def word_overlap(a, b):
    aw = set(re.findall(r'[a-z0-9]+', (a or '').lower()))
    bw = set(re.findall(r'[a-z0-9]+', (b or '').lower()))
    if not aw or not bw: return 0.0
    return len(aw & bw) / len(aw | bw)

conn = duckdb.connect('db/kg2_lexical_store.duckdb', read_only=True)
vocab = VocabularyRetriever(conn)
gold_path = _first_existing(GOLD_CANDIDATES, 'gold')
gold_rows = load_gold(gold_path, NOTE_IDS)
gold_by_note = collections.defaultdict(list)
for g in gold_rows:
    gold_by_note[g['note_id']].append(g)

placeholders = ','.join('?' * len(NOTE_IDS))
rows = conn.execute(f'''
    SELECT e.entity_id, e.note_id, e.orig_start, e.orig_end, e.entity_label, e.original_text,
           n.candidates, r.models
    FROM mollm_review_decisions r
    JOIN extracted_entities e ON e.entity_id = r.entity_id
    JOIN normalized_entities n ON n.entity_id = r.entity_id
    WHERE r.is_test = TRUE AND r.error IS NULL AND r.note_id IN ({placeholders})
''', NOTE_IDS).fetchall()

targets = []
for entity_id, note_id, orig_start, orig_end, entity_label, original_text, cands_json, models_json in rows:
    candidates = json.loads(cands_json) if isinstance(cands_json, str) else (cands_json or [])
    if not candidates:
        continue
    top1 = candidates[0]
    score = top1.get('similarity_score')
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = None
    if score is None or score < 0.75:
        continue
    gold = gold_by_note.get(note_id, [])
    overlapping = [g for g in gold if overlaps(orig_start, orig_end, g['start'], g['end'])]
    if not overlapping:
        continue
    gold_codes = {g['concept_id'] for g in overlapping}
    top1_code = vocab.snomed_code_for_concept(top1.get('omop_concept_id'))
    ground_truth = 'CORRECT' if (top1_code and top1_code in gold_codes) else 'WRONG'
    if ground_truth != 'WRONG':
        continue
    models = json.loads(models_json) if isinstance(models_json, str) else (models_json or [])
    verdicts = [m.get('assessment') for m in models]
    judgment = ensemble_judgment(verdicts)
    if judgment != 'CONFIRMS':
        continue
    overlap = word_overlap(original_text, top1.get('concept_name'))
    targets.append({
        'entity_id': entity_id, 'note_id': note_id, 'entity_label': entity_label,
        'original_text': original_text, 'top1_concept_name': top1.get('concept_name'),
        'top1_score': score, 'top1_basis': top1.get('match_basis'),
        'gold_span': overlapping[0].get('span'), 'gold_concept_id': list(gold_codes)[0],
        'old_verdicts': verdicts, 'name_word_overlap': round(overlap, 2),
    })

print(f'total FN candidates (score>=0.75): {len(targets)}')
# rank: prioritize genuinely divergent names (low word overlap = real semantic
# trap, not a trivially-identical-string crosswalk artifact), among those with
# highest Stage2 confidence (the anchoring-bias regime we're testing).
targets.sort(key=lambda t: (t['name_word_overlap'], -t['top1_score']))
sample = targets[:20]
for t in sample:
    print(f"{t['entity_id']} | {t['note_id']} | {t['entity_label']} | {t['original_text']!r} -> "
          f"{t['top1_concept_name']!r} (score={t['top1_score']}, basis={t['top1_basis']}, "
          f"overlap={t['name_word_overlap']}) | gold: {t['gold_span']!r}")

import os
os.makedirs('reports/contradiction_detection', exist_ok=True)
with open('reports/contradiction_detection/anchoring_bias_targets.json', 'w') as fh:
    json.dump(sample, fh, indent=2, default=str)
conn.close()
