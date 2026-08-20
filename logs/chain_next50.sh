#!/bin/bash
cd /home/ec2-user/clinical_neuro_symbolic_pipeline_reorder

echo "=== CHAIN WATCHER START $(date), waiting for run_batch50.sh to finish ===" >> logs/chain_watcher.log

# Poll until the currently-running batch-1 script process is gone.
while pgrep -f "logs/run_batch50.sh$" > /dev/null; do
    sleep 60
done

echo "=== batch 1 confirmed finished $(date), selecting next 50 fresh notes (excluding calibrator training notes) ===" >> logs/chain_watcher.log

NOTEIDS=$(python3 -c "
import duckdb, pandas as pd
from src.mollm_tier_calibrator import ConsensusCalibrator
conn = duckdb.connect('db/kg2_lexical_store.duckdb', read_only=True)
done = set(conn.execute('SELECT DISTINCT note_id FROM normalized_entities').fetchdf()['note_id'].tolist())
cal = ConsensusCalibrator.load('models/consensus_calibrator_v1.pkl')
leakage = set(cal.training_note_ids or [])
excluded = done | leakage
df = pd.read_csv('data/raw_notes/gold_notes.csv')
fresh = [n for n in df['note_id'].tolist() if n not in excluded]
print(','.join(fresh[:50]))
")

if [ -z "$NOTEIDS" ]; then
    echo "=== no fresh, non-leaked gold-annotated notes left, nothing to chain -- $(date) ===" >> logs/chain_watcher.log
    exit 0
fi

echo "=== next 50 note_ids (leakage-clean): $NOTEIDS ===" >> logs/chain_watcher.log

cat > logs/run_batch50_run2.sh <<INNEREOF
#!/bin/bash
cd /home/ec2-user/clinical_neuro_symbolic_pipeline_reorder
echo "=== BATCH50_RUN2 START \$(date) ===" >> logs/batch50_run2.log
python3 scripts/test_pipeline_e2e.py --note-ids "$NOTEIDS" >> logs/batch50_run2.log 2>&1
echo "=== STAGE1-2B DONE \$(date) ===" >> logs/batch50_run2.log
python3 scripts/run_stage3_tier_gate.py --note-ids "$NOTEIDS" >> logs/batch50_run2.log 2>&1
echo "=== BATCH50_RUN2 FULLY DONE \$(date) ===" >> logs/batch50_run2.log
INNEREOF
chmod +x logs/run_batch50_run2.sh

setsid nohup logs/run_batch50_run2.sh < /dev/null > /dev/null 2>&1 &
echo "=== launched run_batch50_run2.sh, pid $! -- $(date) ===" >> logs/chain_watcher.log
