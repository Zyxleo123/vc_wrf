"""Generated-population WRF and MMD losses for scVIDR, matching the CellFlow definitions.

Generated cells use scVIDR's own dose rule from controls and the drug's 10 µM cells;
the true cells of the sampled condition enter only the target statistic.
"""
import numpy as np
import torch


def response_score_torch(x, program_idx, mu0, s0):
    p = torch.clamp(x[:, program_idx], min=0.)
    shift = p.max(dim=1, keepdim=True)[0].detach()
    scaled = torch.exp(-shift[:, 0]) + (torch.exp(p - shift) * (-torch.expm1(-p))).sum(dim=1)
    return (mu0 - (shift[:, 0] + torch.log(scaled))) / (s0 + 1e-6)


def soft_fractions(x, program_idx, mu0, s0, taus, h):
    r = response_score_torch(x, program_idx, mu0, s0)
    return torch.sigmoid((taus[None, :] - r[:, None]) / h).mean(dim=0)


def mmd2_torch(a, b, mean, components, sigmas):
    za, zb = (a - mean) @ components.T, (b - mean) @ components.T

    def kernel(x, y):
        d = torch.clamp((x * x).sum(1)[:, None] + (y * y).sum(1)[None, :] - 2 * x @ y.T, min=0.)
        return torch.exp(-d[None, :, :] / (2 * sigmas[:, None, None] ** 2)).mean(dim=0)

    aa, bb, ab = kernel(za, za), kernel(zb, zb), kernel(za, zb)
    return (aa.sum() - torch.trace(aa)) / (len(a) * (len(a) - 1)) + (bb.sum() - torch.trace(bb)) / (len(b) * (len(b) - 1)) - 2 * ab.mean()


def attach(model, train, spec, options, config, seed, progress, log):
    module, original = model.module, model.module.loss
    obs = train.obs
    X = torch.from_numpy(np.ascontiguousarray(train.X))
    controls = np.flatnonzero(obs['split'].eq('control').to_numpy())
    treated = obs[obs['split'].eq('train')]
    conditions = sorted(treated['condition'].unique())
    members = {c: np.flatnonzero(obs['condition'].eq(c).to_numpy()) for c in conditions}
    dose = {c: float(treated.loc[treated['condition'].eq(c), 'dose'].iloc[0]) for c in conditions}
    top = {c: np.flatnonzero((obs['drug'].eq(treated.loc[treated['condition'].eq(c), 'drug'].iloc[0]) & obs['dose'].eq(config['max_dose'])).to_numpy()) for c in conditions}
    assert len(conditions) == 78 and all(len(v) > 0 for v in top.values())
    rng = np.random.default_rng(20000 + seed)
    every, begin, n, weight, h = config['auxiliary_every'], config['auxiliary_start_epoch'], config['auxiliary_batch_size'], options['weight'], options['h']
    state = dict(step=0, cache=None)
    log.write('step,epoch,condition,auxiliary_loss,weighted_auxiliary_loss\n')

    def constants(device):
        if state['cache'] is None or state['cache']['device'] != device:
            state['cache'] = dict(device=device, X=X.to(device), idx=torch.as_tensor(spec.program_idx, device=device), taus=torch.as_tensor(config['training_taus'], dtype=torch.float64, device=device), mean=torch.as_tensor(spec.pca_mean, dtype=torch.float32, device=device), components=torch.as_tensor(spec.pca_components, dtype=torch.float32, device=device), sigmas=torch.as_tensor(spec.mmd_sigmas, dtype=torch.float32, device=device))
        return state['cache']

    def loss(tensors, inference_outputs, generative_outputs, **kwargs):
        recorder = original(tensors, inference_outputs, generative_outputs, **kwargs)
        if not module.training:
            return recorder
        state['step'] += 1
        if progress['epoch'] < begin or state['step'] % every:
            return recorder
        k = constants(tensors['X'].device)
        c = conditions[rng.integers(len(conditions))]

        def take(ids):
            return k['X'][torch.as_tensor(rng.choice(ids, min(n, len(ids)), replace=False), device=k['X'].device)]

        real, source, base, high = take(members[c]), take(controls), take(controls), take(top[c])
        module.eval()
        try:
            delta = module.inference(high)['qz_m'].mean(0) - module.inference(base)['qz_m'].mean(0)
            z = module.inference(source)['qz_m'] + delta * float(np.log1p(dose[c]) / np.log1p(config['max_dose']))
            generated = torch.clamp(module.generative(z)['px'], min=0.)
        finally:
            module.train()
        if options['method'] == 'wrf':
            fp = soft_fractions(generated.double(), k['idx'], spec.mu0, spec.s0, k['taus'], h)
            ft = soft_fractions(real.double(), k['idx'], spec.mu0, spec.s0, k['taus'], h).detach()
            aux = ((fp - ft) ** 2).mean().float()
        else:
            aux = mmd2_torch(generated, real.detach(), k['mean'], k['components'], k['sigmas'])
        recorder._loss['auxiliary'] = weight * every * aux
        log.write(f"{state['step']},{progress['epoch']},{c},{float(aux)},{float(weight * every * aux)}\n")
        return recorder

    module.loss = loss
