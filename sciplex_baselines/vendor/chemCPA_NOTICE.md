# chemCPA source provenance

`chemcpa_model.py` is vendored from
[theislab/chemCPA](https://github.com/theislab/chemCPA/blob/a4a4ded0c3b949c64ff1ea51033be1b7c301c36b/chemCPA/model.py),
commit `a4a4ded0c3b949c64ff1ea51033be1b7c301c36b` (2022-12-08),
upstream file blob `18328b6eba8b743026fc1e1db29a4063af57378d`.
The upstream MIT license is reproduced in `chemCPA_LICENSE`.

Reference: Hetzel et al., *Predicting Cellular Responses to Novel Drug
Perturbations at a Single-Cell Resolution*, NeurIPS 2022.

Local fixes: register covariate networks/embeddings as `ModuleList` so evaluation,
device transfers, and checkpoints include them; use `squeeze(-1)` in the amortized
doser to support a single prediction; mask zero-dose effects to zero, as in the
upstream one-hot MLP doser.

Local extension (opt-in, `loss_ae="nb"`; the default `"gauss"` path is unchanged):
the paper-era file defines CPA's `NBLoss` but always trains with
`GaussianNLLLoss`. With `"nb"`, the decoder's first half is a softmax over
features multiplied by the cell's observed library size (the NB mean), its
second half is the NB inverse dispersion, and `update` scores raw `counts`
with `NBLoss`. This follows the count-likelihood design of later CPA releases
(scVI-style decoder with observed library size).

Local hook: `update(..., aux_loss=callable)` adds the callable's value to the
autoencoder objective on autoencoder steps (used for the project's
response-fraction loss; without it, training is unchanged). No other upstream
model changes.

The adapter trains from scratch with frozen RDKit Morgan fingerprints (radius 2,
2048 bits), rather than the paper's pretrained molecular embeddings. It uses a
smaller configurable network, log1p(dose in micromolar) for the amortized doser,
one constant A549 covariate, and the project's fixed expression genes and splits.
The NB variant adds one feature holding all other human UMIs, so generated
cells are normalized by their generated total exactly as real cells are; the
source control cell supplies the library size (treatment-induced library-size
shifts are not modelled).
This is a chemCPA-Morgan baseline, not a reproduction of the published benchmark
or LINCS pretraining. The WRF loss is not included.
