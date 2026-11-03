"""Copy verified results into the existing team directory without exposing private files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

def sha256(path):
    value = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

def package(run, destination):
    run, destination = Path(run), Path(destination)
    assert json.loads((run / 'checks/artifact_validation.json').read_text())['passed']
    selection = json.loads((run / 'selected_methods.json').read_text())['methods']
    assert not destination.exists(), 'Use a fresh directory; do not overwrite a shared package.'
    destination.mkdir(parents=True)
    for folder in ['report', 'metrics', 'diagnostics', 'reference_predictions', 'final_source', 'eval_spec_v1', 'checks']:
        shutil.copytree(run / folder, destination / folder)
    for file in run.iterdir():
        if file.is_file() and file.suffix in {'.json', '.csv', '.txt'}:
            shutil.copy2(file, destination / file.name)
    model_index = []
    for trial in sorted((run / 'trials').iterdir()):
        for directory in sorted(trial.glob('seed_*')):
            relative = directory.relative_to(run)
            target = destination / relative
            target.mkdir(parents=True)
            selected = trial.name in selection.values() and directory.name in {'seed_0', 'seed_1', 'seed_2'}
            for file in directory.iterdir():
                if file.is_file() and (selected or file.suffix not in {'.pkl', '.h5ad'}):
                    shutil.copy2(file, target / file.name)
            if selected:
                method = next(k for k, v in selection.items() if v == trial.name)
                checkpoint = json.loads((target / 'selected_checkpoint.json').read_text())
                model_index.append(dict(method=method, seed=int(directory.name.split('_')[-1]),
                    training_step=checkpoint['step'], model=str(relative / 'best_CellFlow.pkl'),
                    validation_predictions=str(relative / 'preds_val.h5ad'), test_predictions=str(relative / 'preds_test.h5ad')))
    (destination / 'model_index.json').write_text(json.dumps(model_index, indent=2) + '\n')
    root = Path(os.environ['RESPONSE_FRACTION_ROOT'])
    evaluator = destination / 'evaluator'
    evaluator.mkdir()
    with tempfile.TemporaryFile() as stream:
        subprocess.run(['git', '-C', str(root / 'src/vc_wrf_1d8b173'), 'archive',
                        '1d8b173f8d4120c8ba4487d7e03511b26e7a71ad'], stdout=stream, check=True)
        stream.seek(0)
        with tarfile.open(fileobj=stream) as archive:
            archive.extractall(evaluator, filter='data')
    original_hashes = json.loads((run / 'final_artifact_sha256.json').read_text())
    hashes = {}
    for file in sorted(destination.rglob('*')):
        if file.is_file():
            relative = str(file.relative_to(destination))
            hashes[relative] = sha256(file)
            if relative in original_hashes:
                assert hashes[relative] == original_hashes[relative], relative
    (destination / 'shared_bundle_sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    (destination / 'README.txt').write_text(
        'CellFlow v1 results package\n\n'
        'Start with report/cellflow-v1-report.txt and report/test_metrics.png.\n'
        'All eight metrics: report/all_metric_summaries.csv.\n'
        'Detailed results: metrics/ and diagnostics/.\n'
        'Nine selected models and their validation/test predictions: model_index.json.\n'
        'Exact saved source, final configuration, and PSC rerun launcher: final_source/.\n'
        'Frozen unmodified metric repository: evaluator/. Exact dependencies: requirements.lock.txt.\n'
        'Use the shared input panel at /ocean/projects/cis260266p/shared/response-fraction/data/processed/a549_hdaci_v1.h5ad.\n'
        'The benchmark is log1p(CP10K) over the frozen genes; do not renormalize the selected panel.\n\n'
        'This package includes every selected model/prediction, all candidate training records,\n'
        'and the evaluation artifacts. Unselected model weights and interrupted attempts remain\n'
        'in the full original experiment: ' + str(run) + '\n'
        'artifact_inventory.csv inventories that full original experiment, including files not copied here.\n'
        'final_artifact_sha256.json records original artifacts; shared_bundle_sha256.json verifies copied files.\n'
        'The original experiment is preserved privately; edits to these shared copies do not change it.\n\n'
        'Access is limited to ywang179 and amotwani. No other teammate has been added.\n')
    subprocess.run(['setfacl', '-R', '-m', 'u:amotwani:rwX,u:ywang179:rwX,g::---,o::---', str(destination)], check=True)
    directories = [str(destination)] + [str(p) for p in destination.rglob('*') if p.is_dir()]
    for start in range(0, len(directories), 100):
        subprocess.run(['setfacl', '-d', '-m', 'u:amotwani:rwx,u:ywang179:rwx,g::---,o::---',
                        *directories[start:start + 100]], check=True)
    result = dict(status='complete', path=str(destination), selected_models=len(model_index),
                  hashed_files=len(hashes), copied_bytes=sum(p.stat().st_size for p in destination.rglob('*') if p.is_file()),
                  original_hashes_verified=True, access=['ywang179', 'amotwani'])
    (run / 'shared_results_package.json').write_text(json.dumps(result, indent=2) + '\n')
    print('SHARED_PACKAGE_COMPLETE', json.dumps(result), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    args = parser.parse_args()
    package(args.run, args.destination)
