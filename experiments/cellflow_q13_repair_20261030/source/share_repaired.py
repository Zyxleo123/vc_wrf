"""Copy verified results into the existing two-person team results directory."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
root=a.root
destination=Path('/ocean/projects/cis260266p/shared/response-fraction/results/cellflow_q13_precision_repair_20261030')
assert json.loads((root/'complete.json').read_text())['full_three_seed_comparison']

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def included(path):
    relative=path.relative_to(root)
    return not any(part in {'cache','__pycache__','vc_wrf_1d8b173'} for part in relative.parts) and path.name not in {'share.log','artifact_sha256.json','shared_package.json'}

files=[f for f in sorted(root.rglob('*')) if f.is_file() and included(f)]
hashes={str(f.relative_to(root)):sha(f) for f in files}
(root/'artifact_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
destination.mkdir(mode=0o700,exist_ok=False)
for f in [*files,root/'artifact_sha256.json']:
    target=destination/f.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(f,target)
for name,value in hashes.items():assert sha(destination/name)==value,name
subprocess.run(['setfacl','-R','-m','u:ywang179:rwX,u:amotwani:rwX,g::---,o::---',str(destination)],check=True)
for directory in [destination,*[d for d in destination.rglob('*') if d.is_dir()]]:
    subprocess.run(['setfacl','-m','d:u::rwx,d:u:ywang179:rwx,d:u:amotwani:rwx,d:g::---,d:m::rwx,d:o::---',str(directory)],check=True)
metadata=dict(path=str(destination),copied_file_hashes_verified=len(hashes),access=['ywang179','amotwani'],
    original_run=str(root),reference_run='/ocean/projects/cis260266p/shared/response-fraction/results/cellflow_vc_wrf_v1_50003709',
    excluded=['runtime caches','evaluator symlink (pinned evaluator and baseline already archived)','active share log'],no_github_push=True)
(root/'shared_package.json').write_text(json.dumps(metadata,indent=2)+'\n')
shutil.copyfile(root/'shared_package.json',destination/'shared_package.json')
print(json.dumps(metadata),flush=True)
