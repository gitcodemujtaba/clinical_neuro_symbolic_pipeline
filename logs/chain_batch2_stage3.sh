#!/bin/bash
cd /home/ec2-user/clinical_neuro_symbolic_pipeline_reorder

echo "=== BATCH2 STAGE3 WATCHER (v2, correct process names) START $(date) ===" >> logs/chain_watcher.log

while pgrep -f "logs/run_batch2_stage12b.sh$" > /dev/null; do
    sleep 60
done
echo "=== batch2 Stage1-2b confirmed finished $(date) ===" >> logs/chain_watcher.log

while pgrep -f "logs/run_batch50_resume.sh$" > /dev/null; do
    sleep 60
done
echo "=== batch1 Stage3 (resumed) confirmed finished $(date), launching batch2 Stage3 ===" >> logs/chain_watcher.log

cat > logs/run_batch2_stage3.sh <<INNEREOF
#!/bin/bash
cd /home/ec2-user/clinical_neuro_symbolic_pipeline_reorder
echo "=== BATCH2 STAGE3 START \$(date) ===" >> logs/batch50_run2.log
python3 scripts/run_stage3_tier_gate.py --note-ids "11524757-DS-4,11576109-DS-15,11652327-DS-14,11654232-DS-17,11684467-DS-26,11714071-DS-56,11783215-DS-18,11799619-DS-27,11806511-DS-10,11810606-DS-7,11834909-DS-2,11838076-DS-20,11855597-DS-22,11859945-DS-29,11891099-DS-13,11903286-DS-2,11922236-DS-28,12018901-DS-68,12050253-DS-20,12093726-DS-4,12101085-DS-27,12135369-DS-23,12152043-DS-10,12190214-DS-14,12204158-DS-10,12204513-DS-26,12206678-DS-17,12276520-DS-22,12286594-DS-21,12286975-DS-22,12290802-DS-3,12298181-DS-9,12298967-DS-22,12304719-DS-18,12314513-DS-16,12319089-DS-12,12340122-DS-5,12407834-DS-17,12412316-DS-14,12431768-DS-17,12465457-DS-18,12484093-DS-33,12515572-DS-13,12549331-DS-3,12574098-DS-24,12582583-DS-20,12612379-DS-23,12618758-DS-3,12626414-DS-28,12626414-DS-32" >> logs/batch50_run2.log 2>&1
echo "=== BATCH2 FULLY DONE \$(date) ===" >> logs/batch50_run2.log
INNEREOF
chmod +x logs/run_batch2_stage3.sh
setsid nohup logs/run_batch2_stage3.sh < /dev/null > /dev/null 2>&1 &
echo "=== launched run_batch2_stage3.sh, pid $! -- $(date) ===" >> logs/chain_watcher.log
