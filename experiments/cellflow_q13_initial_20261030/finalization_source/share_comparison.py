"""Archive the completed attempts with the existing two-person team permissions."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--run', required=True, type=Path)
p.add_argument('--destination', required=True, type=Path)
a = p.parse_args()
assert (a.run / 'complete.json').is_file()

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

# The report process has exited, so its log is now stable as well.
hashes = {str(f.relative_to(a.run)): sha(f) for f in sorted(a.run.rglob('*'))
          if f.is_file() and f.name not in {'artifact_sha256.json', 'shared_package.json'}}
(a.run / 'artifact_sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
a.destination.mkdir(mode=0o700, parents=False, exist_ok=False)
shutil.copytree(a.run, a.destination, dirs_exist_ok=True)
for name, value in hashes.items():
    assert sha(a.destination / name) == value, name
assert sha(a.destination / 'artifact_sha256.json') == sha(a.run / 'artifact_sha256.json')
subprocess.run(['setfacl', '-R', '-m', 'u:ywang179:rwX,u:amotwani:rwX,g::---,o::---', str(a.destination)], check=True)
for directory in [a.destination, *[f for f in a.destination.rglob('*') if f.is_dir()]]:
    subprocess.run(['setfacl', '-m', 'd:u::rwx,d:u:ywang179:rwx,d:u:amotwani:rwx,d:g::---,d:m::rwx,d:o::---', str(directory)], check=True)
metadata = dict(path=str(a.destination), copied_file_hashes_verified=len(hashes),
    access=['ywang179', 'amotwani'], original_run=str(a.run),
    shared_reference_run='/ocean/projects/cis260266p/shared/response-fraction/results/cellflow_vc_wrf_v1_50003709')
(a.run / 'shared_package.json').write_text(json.dumps(metadata, indent=2) + '\n')
shutil.copyfile(a.run / 'shared_package.json', a.destination / 'shared_package.json')
print(json.dumps(metadata), flush=True)
