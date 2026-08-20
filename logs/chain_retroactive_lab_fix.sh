#!/bin/bash
cd /home/ec2-user/clinical_neuro_symbolic_pipeline_reorder
LOG=logs/retroactive_lab_fix.log

echo "=== RETROACTIVE LAB-FIX WATCHER (v2, includes fresh grading) START $(date), waiting for batch2 Stage3 to finish ===" >> "$LOG"

while pgrep -f "logs/run_batch2_stage3.sh$" > /dev/null; do
    sleep 60
done
echo "=== batch2 Stage3 confirmed finished $(date), running retroactive_lab_procedure_fix.py ===" >> "$LOG"

python3 scripts/retroactive_lab_procedure_fix.py >> "$LOG" 2>&1
echo "=== retroactive fix script done $(date) ===" >> "$LOG"

NOTEIDS=$(tail -30 "$LOG" | grep -A1 "notes touched" | tail -1)

if [ -z "$NOTEIDS" ] || [[ "$NOTEIDS" == *"notes touched"* ]]; then
    echo "=== no note_ids captured, nothing to re-run Stage 3 for -- $(date) ===" >> "$LOG"
else
    echo "=== re-running Stage 3 for affected notes: $NOTEIDS -- $(date) ===" >> "$LOG"
    python3 scripts/run_stage3_tier_gate.py --note-ids "$NOTEIDS" >> "$LOG" 2>&1
    echo "=== affected-notes Stage 3 re-run done $(date) ===" >> "$LOG"
fi

echo "=== running fresh full-corpus grading (post-fix) $(date) ===" >> "$LOG"
python3 <<'PYEOF' >> "$LOG" 2>&1
from src.db_utils import connect_with_retry
from evaluation.tier_gate_grading import grade_by_tier

conn = connect_with_retry('db/kg2_lexical_store.duckdb', read_only=True, max_wait_seconds=1800)
all_notes = sorted(r[0] for r in conn.execute('SELECT DISTINCT note_id FROM mollm_tier_gate_decisions').fetchall())
print(f"\n=== POST-FIX FULL-CORPUS GRADING: {len(all_notes)} notes ===")
report = grade_by_tier(conn, all_notes)
for tier, d in report.items():
    c = d['clean']
    prec = f"{c['precision']*100:.1f}%" if c['precision'] is not None else "n/a"
    print(f"{tier:35s} n_decisions={d['n_decisions']:5d}  clean_n={c['n']:5d}  clean_precision={prec}")
conn.close()
PYEOF

echo "=== RETROACTIVE FIX + FRESH GRADING FULLY DONE $(date) ===" >> "$LOG"
