#!/bin/bash
# Monitor experiment progress by checking cache counts
cd /home/bsada1/coderepo/MedMCQA-Robustness-Study
python3 -c "
import sqlite3, time
conn = sqlite3.connect('outputs/cache/responses.db')
c = conn.cursor()
c.execute('SELECT model_name, experiment, COUNT(*) FROM cache GROUP BY model_name, experiment ORDER BY model_name, experiment')
print(f'[{time.strftime(\"%Y-%m-%d %H:%M:%S\")}] Cache Progress:')
for row in c.fetchall():
    print(f'  {row[0]:30s} {row[1]:40s} {row[2]:>6d}')
conn.close()
"
echo ""
echo "Running processes:"
ps aux | grep "run_experiment\|main.py" | grep python | grep -v grep | awk '{print "  PID " $2 ": " $11 " " $12 " " $13 " " $14 " " $15}'
echo ""
echo "GPU usage:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
