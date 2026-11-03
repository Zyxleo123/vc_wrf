"""Train and generate the matched 3- and 13-threshold WRF variants."""
import argparse
import json
from pathlib import Path
import shutil

TRIAL = 'wrf_l0p1_h0p3'


def initialize(root):
    import run_cellflow_v1 as base
    package = Path(__file__).resolve().parents[1]
    root.mkdir(parents=True, exist_ok=False)
    for count in [3, 13]:
        run = root / f'run{count}'
        run.mkdir()
        shutil.copyfile(package / f'config_{count}.json', run / 'config.json')
        config = json.loads((run / 'config.json').read_text())
        method = 'cellflow_wrf' if count == 3 else 'cellflow_wrf13'
        base.save_json(run / 'selected_methods.json', {'methods': {method: TRIAL}})
        base.audit_panel(run)
        expected = json.loads((base.REPO / 'docs/cellflow/artifacts.json').read_text())['panel_sha256']
        assert json.loads((run / 'panel_audit.json').read_text())['panel_sha256'] == expected
        shutil.copytree(base.REPO / 'artifacts/eval_spec_v1', run / 'eval_spec_v1')
        assert config['seeds'] == [0, 1, 2]


def train(root, count, seed):
    import jax
    import jax.numpy as jnp
    import run_cellflow_v1 as base
    import cellflow_population_loss as population
    run = root / f'run{count}'
    config = json.loads((run / 'config.json').read_text())
    original = population.make_population_step
    checks = 0

    @jax.jit
    def all_finite(tree):
        return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(tree)]))

    def instrument(solver, *args, **kwargs):
        update = original(solver, *args, **kwargs)

        def step(rng, batch):
            nonlocal checks
            values = update(rng, batch)
            if bool(values[3]):
                assert bool(all_finite((solver.vf_state.params, solver.vf_state.opt_state)))
                checks += 1
            return values
        return step

    population.make_population_step = instrument
    try:
        base.train_seed(run, seed, TRIAL)
    finally:
        population.make_population_step = original
    expected = (config['training_steps'] - config['auxiliary_start_step']) // config['auxiliary_every']
    assert checks == expected
    base.save_json(run / 'trials' / TRIAL / f'seed_{seed}' / 'numerical_check.json',
                   dict(passed=True, finite_state_checks=checks))


def predict(root):
    import anndata as ad
    from cellflow.model import CellFlow
    import run_cellflow_v1 as base
    frozen = {}
    for count in [3, 13]:
        run = root / f'run{count}'
        assert json.loads((run / 'validation_integration_audit.json').read_text())['passed']
        for seed in [0, 1, 2]:
            directory = run / 'trials' / TRIAL / f'seed_{seed}'
            assert json.loads((directory / 'training_complete.json').read_text())['steps'] == 30000
            assert json.loads((directory / 'numerical_check.json').read_text())['passed']
            frozen[f'{count}/{seed}'] = dict(
                sha256=base.digest(directory / 'best_CellFlow.pkl'),
                checkpoint=json.loads((directory / 'selected_checkpoint.json').read_text()))
    freeze_path = root / 'selected_models.json'
    if freeze_path.exists():
        assert json.loads(freeze_path.read_text()) == frozen
    else:
        base.save_json(freeze_path, frozen)
    panel = ad.read_h5ad(base.PANEL)
    train_data = base.model_data(panel[panel.obs['split'].isin(['train', 'control'])])
    entries = []
    for count in [3, 13]:
        run = root / f'run{count}'
        base.save_json(run / 'test_evaluation_started.json', frozen)
        config = json.loads((run / 'config.json').read_text())
        for seed in [0, 1, 2]:
            directory = run / 'trials' / TRIAL / f'seed_{seed}'
            model_path = directory / 'best_CellFlow.pkl'
            assert base.digest(model_path) == frozen[f'{count}/{seed}']['sha256']
            output = directory / 'preds_test.h5ad'
            if not output.exists():
                model = CellFlow.load(str(model_path))
                prediction = base.predict(model, train_data, config, 'test', seed)
                temporary = directory / 'preds_test.tmp.h5ad'
                prediction.write_h5ad(temporary, compression='gzip')
                temporary.rename(output)
                del model, prediction
            for split in ['val', 'test']:
                path = directory / f'preds_{split}.h5ad'
                entries.append(dict(method='cellflow_wrf' if count == 3 else 'cellflow_wrf13',
                                    seed=seed, split=split, path=str(path.resolve()), sha256=base.digest(path)))
    base.save_json(root / 'artifacts.json', dict(panel_sha256=base.digest(base.PANEL), predictions=entries))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['initialize', 'train', 'predict'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--thresholds', type=int, choices=[3, 13])
    parser.add_argument('--seed', type=int, choices=[0, 1, 2])
    args = parser.parse_args()
    if args.stage == 'initialize':
        initialize(args.root)
    elif args.stage == 'train':
        if args.thresholds is None or args.seed is None:
            parser.error('train requires --thresholds and --seed')
        train(args.root, args.thresholds, args.seed)
    else:
        predict(args.root)


if __name__ == '__main__':
    main()
