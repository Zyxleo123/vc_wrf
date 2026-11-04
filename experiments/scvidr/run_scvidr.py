"""Train scVIDR on the frozen v1 panel and generate held-out-dose populations."""
import argparse
import copy
import hashlib
import json
import os
import shutil
from pathlib import Path
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ['RESPONSE_FRACTION_DATA'])
SCVIDR = Path(os.environ['SCVIDR_SOURCE'])
PANEL = DATA / 'processed/a549_hdaci_v1.h5ad'
sys.path.insert(0, str(SCVIDR / 'vidr'))
sys.path.insert(0, str(REPO))
from sciplex_eval import EvalSpec, evaluate_predictions, summarize


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def model_data(a):
    a = a.copy()
    a.X = (a.X.toarray() if sparse.issparse(a.X) else np.asarray(a.X)).astype(np.float32)
    a.obs['cell_type'] = 'A549'
    a.obs['drug_dose'] = a.obs['drug'].astype(str) + '_' + a.obs['dose'].astype(str)
    return a


def predict(model, train, config, split_name, seed):
    import torch
    split = json.loads((REPO / 'configs/split_v1.json').read_text())
    selected = [c for c in split['conditions'] if c['split'] == split_name]
    n = config['prediction_cells_per_condition']
    model.module.eval()
    latent = model.get_latent_representation(train)
    controls = train.obs['split'].eq('control').to_numpy()
    control_ids = np.flatnonzero(controls)
    control_mean = latent[controls].mean(0)
    rng = np.random.default_rng(10000 + seed)
    arrays, frames, negative, elements = [], [], 0, 0
    for c in selected:
        top = (train.obs['drug'].eq(c['drug']) & train.obs['dose'].eq(config['max_dose'])).to_numpy()
        assert top.sum() > 0 and not train.obs['condition'].eq(c['condition']).any(), c['condition']
        delta = latent[top].mean(0) - control_mean
        source = rng.choice(control_ids, n, replace=False)
        z = latent[source] + delta * np.log1p(c['dose_nM']) / np.log1p(config['max_dose'])
        with torch.no_grad():
            raw = model.module.generative(torch.Tensor(z))['px'].cpu().numpy().astype(np.float32)
        assert raw.shape == (n, train.n_vars) and np.isfinite(raw).all(), (c['condition'], raw.shape)
        negative += int((raw < 0).sum()); elements += int(raw.size)
        arrays.append(np.maximum(raw, 0))
        frames.append(pd.DataFrame({'condition': c['condition'], 'source_control_id': train.obs_names[source], 'n_delta_cells': int(top.sum())}, index=[f"{c['condition']}|generated_{i}" for i in range(n)]))
    var = pd.DataFrame(index=pd.Index(list(train.var_names)))
    pred = ad.AnnData(np.vstack(arrays), obs=pd.concat(frames), var=var)
    pred.uns['normalization'] = 'log1p(CP10K), human-gene library size'
    pred.uns['postprocessing'] = config['prediction_postprocessing']
    pred.uns['negative_fraction_before_clipping'] = negative / elements
    for c in selected:
        assert (pred.obs['condition'] == c['condition']).sum() == n >= c['n_cells']
    return pred


def convert(root):
    a = ad.read_h5ad(PANEL)
    ad.settings.allow_write_nullable_strings = False
    for frame in [a.obs, a.var]:
        frame.index = pd.Index(frame.index.astype(object), dtype=object)
        for column in frame.columns:
            if frame[column].dtype.kind in 'OUT' or str(frame[column].dtype) in ['str', 'string']:
                frame[column] = frame[column].astype(object)
    root.mkdir(parents=True, exist_ok=True)
    a.write_h5ad(root / 'panel_legacy.h5ad')
    b = ad.read_h5ad(root / 'panel_legacy.h5ad')
    assert (a.X != b.X).nnz == 0 and (a.layers['counts'] != b.layers['counts']).nnz == 0
    assert a.obs.astype(str).equals(b.obs.astype(str)) and list(a.var_names) == list(b.var_names)
    save_json(root / 'panel_legacy.json', dict(source=str(PANEL), source_sha256=digest(PANEL), legacy_sha256=digest(root / 'panel_legacy.h5ad'), reason='anndata 0.8 in the scVIDR environment cannot read nullable-string-array encoding'))


def load_panel(root):
    record = json.loads((root / 'panel_legacy.json').read_text())
    assert record['source_sha256'] == json.loads((REPO / 'docs/cellflow/artifacts.json').read_text())['panel_sha256']
    assert digest(root / 'panel_legacy.h5ad') == record['legacy_sha256']
    return ad.read_h5ad(root / 'panel_legacy.h5ad')


def train_seed(root, seed):
    CONFIG = root / 'config.json'
    import scvi
    import torch
    from pytorch_lightning.callbacks import Callback
    from scvi.data import setup_anndata
    from vidr import VIDR
    config = json.loads(CONFIG.read_text())
    directory = root / f'seed_{seed}'
    directory.mkdir(parents=True, exist_ok=False)
    scvi.settings.seed = seed
    spec = EvalSpec.load(REPO / 'artifacts/eval_spec_v1')
    panel = load_panel(root)
    assert list(panel.var_names) == spec.gene_ids
    train = model_data(panel[panel.obs['split'].isin(['train', 'control'])])
    val = panel[panel.obs['split'].isin(['val', 'control'])].copy()
    del panel
    assert train.n_obs == 19398 and train.n_vars == 2085
    train.obs[['condition', 'split', 'replicate', 'drug_dose']].to_csv(directory / 'training_cell_ids.csv')
    setup_anndata(train, batch_key='drug_dose', labels_key='cell_type')
    model = VIDR(train, hidden_dim=config['hidden_dim'], latent_dim=config['latent_dim'], n_hidden_layers=config['n_hidden_layers'], dropout_rate=config['dropout_rate'], kl_weight=config['kl_weight'], linear_decoder=config['linear_decoder'])
    states = {}

    class Snapshot(Callback):
        def on_train_epoch_end(self, trainer, module, *args):
            epoch = trainer.current_epoch + 1
            if epoch in config['validation_epochs']:
                states[epoch] = copy.deepcopy(model.module.state_dict())

    start = time.monotonic()
    model.train(max_epochs=config['max_epochs'], batch_size=config['batch_size'], train_size=config['train_size'], early_stopping=config['early_stopping'], early_stopping_patience=config['early_stopping_patience'], callbacks=[Snapshot()])
    elapsed = time.monotonic() - start
    pd.concat(list(model.history.values()), axis=1).to_csv(directory / 'training_history.csv')
    epochs = len(model.history['elbo_train'])
    states[epochs] = copy.deepcopy(model.module.state_dict())
    best, validations = (float('inf'), float('inf')), []
    for epoch in sorted(states):
        model.module.load_state_dict(states[epoch])
        pred = predict(model, train, config, 'val', seed)
        records = evaluate_predictions(spec, val, pred, split='val', seed=0)
        summary = summarize(records)
        score = tuple(summary[k]['mean'] for k in config['checkpoint_selection'])
        selected = score < best
        validations.append(dict(epoch=epoch, score=list(score), selected=selected, negative_fraction=float(pred.uns['negative_fraction_before_clipping']), **{k: v['mean'] for k, v in summary.items() if isinstance(v, dict)}))
        if selected:
            best, best_epoch, best_pred = score, epoch, pred
        print('VALIDATION', seed, epoch, 'primary', score[0], 'w1', score[1], 'selected', selected, flush=True)
    save_json(directory / 'validation_history.json', validations)
    model.module.load_state_dict(states[best_epoch])
    model.save(str(directory / 'model'), overwrite=True)
    best_pred.write_h5ad(directory / 'preds_val.h5ad', compression='gzip')
    save_json(directory / 'selected_checkpoint.json', dict(seed=seed, epoch=best_epoch, epochs_trained=epochs, selection_keys=config['checkpoint_selection'], selection_scores=list(best), selected_on='validation only', model_sha256=digest(directory / 'model/model_params.pt')))
    pred = predict(model, train, config, 'test', seed)
    pred.write_h5ad(directory / 'preds_test.h5ad', compression='gzip')
    save_json(directory / 'training_complete.json', dict(seed=seed, epochs=epochs, elapsed_seconds=elapsed, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, job_id=os.environ.get('SLURM_JOB_ID'), versions={'torch': torch.__version__, 'scvi-tools': scvi.__version__, 'anndata': ad.__version__}))
    print('TRAINING_COMPLETE', seed, best_epoch, flush=True)


def index(root):
    CONFIG = root / 'config.json'
    config = json.loads(CONFIG.read_text())
    cellflow = json.loads((REPO / 'docs/cellflow/artifacts.json').read_text())
    entries = []
    for seed in config['seeds']:
        directory = root / f'seed_{seed}'
        selected = json.loads((directory / 'selected_checkpoint.json').read_text())
        assert digest(directory / 'model/model_params.pt') == selected['model_sha256']
        for split in ['val', 'test']:
            path = directory / f'preds_{split}.h5ad'
            entries.append(dict(method='scvidr', seed=seed, split=split, path=str(path.resolve()), sha256=digest(path)))
    entries += [p for p in cellflow['predictions'] if p['method'] in ['unchanged_control', 'nearest_dose', 'adjacent_dose_mixture']]
    save_json(root / 'artifacts.json', dict(panel_sha256=json.loads((root / 'panel_legacy.json').read_text())['source_sha256'], panel_relative_path='processed/a549_hdaci_v1.h5ad', evaluation_commit=config['evaluation_commit'], predictions=entries))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['convert', 'train', 'index'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=[0, 1, 2])
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if (args.root / 'config.json').exists():
        assert json.loads((args.root / 'config.json').read_text()) == json.loads(args.config.read_text()), 'root was initialized with a different config'
    else:
        shutil.copyfile(args.config, args.root / 'config.json')
    if args.stage == 'convert':
        convert(args.root)
    elif args.stage == 'train':
        if args.seed is None:
            parser.error('train requires --seed')
        train_seed(args.root, args.seed)
    else:
        index(args.root)


if __name__ == '__main__':
    main()
