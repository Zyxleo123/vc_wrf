"""Run a disjoint portion of the fixed training manifest on one allocated GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import re

p=argparse.ArgumentParser()
p.add_argument('--run',type=Path,required=True)
p.add_argument('--phase',choices=['initial','repeat'],required=True)
p.add_argument('--worker',type=int,required=True)
p.add_argument('--workers',type=int,default=3)
a=p.parse_args()
root=Path(os.environ['RESPONSE_FRACTION_ROOT'])
if a.phase=='initial':
 jobs=json.loads((a.run/'initial_jobs.json').read_text())
else:
 selected=json.loads((a.run/'selected_methods.json').read_text())['methods']
 jobs=[dict(trial=trial,seed=seed) for method,trial in selected.items() if method!='cellflow' for seed in [1,2]]
logs=a.run/'logs'; logs.mkdir(exist_ok=True)
for index,job in enumerate(jobs):
 if index%a.workers!=a.worker: continue
 complete=a.run/'trials'/job['trial']/f"seed_{job['seed']}"/'training_complete.json'
 if complete.exists():
  print('SKIP_COMPLETE',job,flush=True);continue
 failed=complete.parent/'training_failed.json'
 if failed.exists() and a.phase=='initial' and job['trial']!='cellflow':
  print('SKIP_FAILED_CANDIDATE',job,flush=True);continue
 log=logs/f"{job['trial']}_seed{job['seed']}.log"
 print('START',job,'log',str(log),flush=True)
 with open(log,'x') as stream:
  result=subprocess.run([sys.executable,'-u',str(root/'src/run_cellflow_v1.py'),'train',
    '--run',str(a.run),'--trial',job['trial'],'--seed',str(job['seed'])],stdout=stream,stderr=subprocess.STDOUT)
 if result.returncode:
  numerical=re.search(r'AssertionError: \((\d+), (nan|inf|-inf)\)',log.read_text())
  if numerical and a.phase=='initial' and job['trial']!='cellflow':
   failed.write_text(json.dumps(dict(status='failed',reason='nonfinite_training_loss',
      step=int(numerical.group(1)),value=numerical.group(2),log=str(log),
      excluded_from_selection=True,test_evaluated=False),indent=2)+'\n')
   print('FAILED_NUMERICAL_CANDIDATE',job,'retained and excluded from selection',flush=True)
   continue
  print('FAILED',job,'exit',result.returncode,flush=True)
  raise SystemExit(result.returncode)
 print('COMPLETE',job,flush=True)
print('WORKER_COMPLETE',a.phase,a.worker,flush=True)
