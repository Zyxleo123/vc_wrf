"""DOLORIS networks for molecular perturbations, ported from the released model.py.

TargetModel is the single network that plays both DDIB diffusion models (paper Sec. 3.7):
  * a control cell (dosage == 0) goes through the shared trunk plus a learned control token and
    decoder_control: the source model, conditioned on cell type only
  * a treated cell goes through the same trunk plus the drug-dose encoding and decoder_treated,
    and the decoder output is added to x_noisy, the control prior (training: control mean +
    control SD * N(0, 1) per gene; inference: the very control cell being translated)
Control and treated rows are run as separate sub-batches, as in the released forward(), so
BatchNorm sees each group on its own.

MaskModel predicts per-gene probabilities of a non-zero value for a drug-dose condition from a
control prior, through a gene graph (GAT over gene nodes). It is trained apart from the diffusion
model with binary cross-entropy.

Only the modules on the molecular code path are built. Unused modules of the released classes
(gene-perturbation GAT in the condition network, cell-type encoders when there is a single cell
type, the mask model's unused dense layers) are left out; they receive no gradient there.
torch_geometric's GATConv is reimplemented (same attention, softmax over incoming edges, self
loops, concatenated heads, initialisation) so that the project environment needs no new package.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class MLP(nn.Module):
    """Linear -> BatchNorm -> SiLU -> Dropout(0.1) for every layer but the last (released MLP)."""

    def __init__(self, sizes, batch_norm=True, last_layer_act="linear", dropout_rate=0.1):
        super().__init__()
        self.batch_norm = batch_norm
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(sizes[i + 1]))
                layers.append(nn.SiLU())
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
        if last_layer_act == "ReLU":
            layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # a single row is duplicated so BatchNorm can run in training mode, then dropped again
        if x.dim() == 3:
            b, l, d = x.shape
            single = b == 1 and self.batch_norm
            if single:
                x, b = x.repeat(2, 1, 1), 2
            x = self.network(x.reshape(b * l, d)).view(b, l, -1)
            return x[:1] if single else x
        single = x.shape[0] == 1 and self.batch_norm
        if single:
            x = x.repeat(2, 1)
        x = self.network(x)
        return x[:1] if single else x


class BlockA(nn.Module):
    """Block_A; the cell-type encoder is built only with several cell types (unused otherwise)."""

    def __init__(self, input_dim, output_dim, time_embed_dim, depth=3, cell_type_embed_dim=None):
        super().__init__()
        self.mlp1 = MLP([input_dim] + depth * [output_dim])
        self.time_encoder = MLP([time_embed_dim] + 2 * [output_dim])
        self.mlp2 = MLP([output_dim] + depth * [output_dim])
        self.mlp3 = MLP((depth + 1) * [output_dim])
        self.cell_type_encoder = MLP([cell_type_embed_dim] + 2 * [output_dim]) if cell_type_embed_dim else None

    def forward(self, x, time_emb, cell_type_emb=None):
        f = self.mlp2(self.mlp1(x) + self.time_encoder(time_emb))
        if cell_type_emb is not None:
            f = f + self.cell_type_encoder(cell_type_emb)
        return self.mlp3(f)


class MoleEncoder(nn.Module):
    """Mole_encoder: MLP of the Uni-Mol embedding, scaled by sigmoid(Dosager(embedding, dose))."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mole_mlp = MLP([input_dim, input_dim, output_dim])
        self.dosager = MLP([output_dim + 1, output_dim, output_dim, 1])

    def forward(self, mole, dosage):
        m = self.mole_mlp(mole)
        return torch.sigmoid(self.dosager(torch.cat([m, dosage], dim=1))) * m


class ConditionNet(nn.Module):
    """GRN_conditional_network on the molecular path: control token or drug-dose code -> decoder."""

    def __init__(self, mole_dim, embed_dim, output_dim, cell_type_embed_dim=None):
        super().__init__()
        self.control_token = nn.Parameter(torch.empty(1, embed_dim))
        nn.init.xavier_normal_(self.control_token)
        self.mole_encoder = MoleEncoder(mole_dim, embed_dim)
        self.decoder = MLP([embed_dim] + 3 * [output_dim])
        if cell_type_embed_dim:  # cell_type_num > 1: output += LayerNorm(cell_type_encoder(cell type embedding))
            self.cell_type_encoder = MLP([cell_type_embed_dim] + 2 * [embed_dim])
            self.ln = nn.LayerNorm(embed_dim)

    def forward(self, n: int | None = None, mole=None, dosage=None, cell_type_emb=None):
        code = self.control_token.repeat(n, 1) if mole is None else self.mole_encoder(mole, dosage)
        if cell_type_emb is not None:
            code = code + self.ln(self.cell_type_encoder(cell_type_emb))
        return self.decoder(code)


class TargetModel(nn.Module):
    def __init__(self, gene_num, hidden_dim=1024, time_pos_dim=1024, embed_dim=256, mole_dim=512, cell_type_num=1,
                 cell_type_embed_dim=256):
        super().__init__()
        self.time_pos_dim = time_pos_dim
        ct_dim = cell_type_embed_dim if cell_type_num > 1 else None  # the released classes add these only then
        self.cell_type_embedding = nn.Embedding(cell_type_num, cell_type_embed_dim) if ct_dim else None
        self.block_1 = BlockA(gene_num, hidden_dim, time_pos_dim, cell_type_embed_dim=ct_dim)
        self.decoder_control = MLP([hidden_dim] + 3 * [2 * hidden_dim] + [gene_num])
        self.decoder_treated = MLP([hidden_dim] + 3 * [2 * hidden_dim] + [gene_num])
        self.control_net = ConditionNet(mole_dim, embed_dim, hidden_dim, cell_type_embed_dim=ct_dim)

    def forward(self, x, timesteps, mole=None, dosage=None, ctrl_mean=None, ctrl_var=None, cell_type=None):
        if ctrl_mean is not None and ctrl_var is not None:
            x_noisy = ctrl_mean + torch.sqrt(ctrl_var) * torch.randn_like(x)
        else:
            x_noisy = ctrl_mean
        temb = timestep_embedding(timesteps, self.time_pos_dim)
        ce = None
        if self.cell_type_embedding is not None:
            if cell_type is None:
                raise ValueError("this model has several cell types: pass cell_type")
            ce = self.cell_type_embedding(cell_type.reshape(-1))
        part = lambda i: None if ce is None else ce[i]
        is_ctrl = dosage.reshape(-1) == 0
        out = torch.zeros_like(x)
        ic, it = torch.nonzero(is_ctrl).squeeze(-1), torch.nonzero(~is_ctrl).squeeze(-1)
        if len(ic):
            cond = self.control_net(n=len(ic), cell_type_emb=part(ic))
            out[ic] = self.decoder_control(self.block_1(x[ic], temb[ic], part(ic)) + cond)
        if len(it):
            if x_noisy is None:
                raise ValueError("treated rows need ctrl_mean (the control prior)")
            cond = self.control_net(mole=mole[it], dosage=dosage[it], cell_type_emb=part(it))
            out[it] = self.decoder_treated(self.block_1(x[it], temb[it], part(it)) + cond) + x_noisy[it]
        return out


# ---- gene graph and mask model ------------------------------------------------------------------
def graph_edges(adjacency: np.ndarray) -> torch.Tensor:
    """Directed edge list (2, E) for a symmetric 0/1 adjacency: off-diagonal edges in both directions
    plus one self loop per node, as GATConv(add_self_loops=True) builds it."""
    a = np.asarray(adjacency, dtype=bool).copy()
    np.fill_diagonal(a, False)
    a |= a.T
    src, dst = np.nonzero(a)
    n = a.shape[0]
    loops = np.arange(n)
    return torch.as_tensor(np.stack([np.concatenate([src, loops]), np.concatenate([dst, loops])]), dtype=torch.long)


class GATConv(nn.Module):
    """torch_geometric.nn.GATConv(in_dim, out_dim, heads, concat=True) with default settings."""

    def __init__(self, in_dim, out_per_head, heads, negative_slope=0.2):
        super().__init__()
        self.heads, self.out = heads, out_per_head
        self.negative_slope = negative_slope
        self.lin = nn.Linear(in_dim, heads * out_per_head, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, out_per_head))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_per_head))
        self.bias = nn.Parameter(torch.zeros(heads * out_per_head))
        nn.init.xavier_uniform_(self.lin.weight)
        bound = math.sqrt(6.0 / (heads + out_per_head))  # PyG glorot on (1, H, C): fan sizes H and C
        nn.init.uniform_(self.att_src, -bound, bound)
        nn.init.uniform_(self.att_dst, -bound, bound)

    def forward(self, x, src, dst):
        m = x.shape[0]
        h = self.lin(x).view(m, self.heads, self.out)
        e = F.leaky_relu((h * self.att_src).sum(-1)[src] + (h * self.att_dst).sum(-1)[dst], self.negative_slope)
        idx = dst[:, None].expand(-1, self.heads)
        e_max = torch.full((m, self.heads), float("-inf"), device=x.device, dtype=e.dtype)
        e_max = e_max.scatter_reduce(0, idx, e, reduce="amax", include_self=True)
        w = torch.exp(e - e_max[dst])
        denom = torch.zeros((m, self.heads), device=x.device, dtype=w.dtype).index_add_(0, dst, w) + 1e-16
        alpha = w / denom[dst]
        out = torch.zeros((m, self.heads, self.out), device=x.device, dtype=h.dtype)
        out.index_add_(0, dst, h[src] * alpha[..., None])
        return out.reshape(m, self.heads * self.out) + self.bias


class MultiLayerGAT(nn.Module):
    """MultiLayerGAT(dim, heads, dropout=0.1, use_norm=True): GAT -> LayerNorm -> ReLU -> dropout."""

    def __init__(self, dims, heads, edges: torch.Tensor, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(GATConv(dims[i], dims[i + 1] // heads[i], heads[i]) for i in range(len(dims) - 1))
        self.norms = nn.ModuleList(nn.LayerNorm(dims[i + 1]) for i in range(len(dims) - 1))
        self.dropout = dropout
        self.register_buffer("edges", edges, persistent=False)

    def forward(self, feature):  # [B, N, D]: B copies of the graph, as Batch.from_data_list builds them
        b, n, d = feature.shape
        offset = (torch.arange(b, device=feature.device) * n)[:, None]
        src = (self.edges[0][None] + offset).reshape(-1)
        dst = (self.edges[1][None] + offset).reshape(-1)
        x = feature.reshape(b * n, d)
        for conv, norm in zip(self.layers, self.norms):
            x = F.dropout(F.relu(norm(conv(x, src, dst))), p=self.dropout, training=self.training)
        return x.view(b, n, -1)


class MaskNet(nn.Module):
    """GRN_conditional_network_for_mask on the molecular path."""

    def __init__(self, gene_num, edges, gene_init_dim=64, mole_dim=512, heads=2, cell_type_embed_dim=None):
        super().__init__()
        self.gene_num = gene_num
        self.cell_type_encoder = MLP([cell_type_embed_dim] + 2 * [gene_init_dim]) if cell_type_embed_dim else None
        self.gene_embedding = nn.Embedding(gene_num, gene_init_dim)
        self.indv_w = nn.Parameter(torch.rand(1, gene_num, gene_init_dim))
        nn.init.xavier_normal_(self.indv_w)
        self.indv_b = nn.Parameter(torch.rand(1, gene_num))
        nn.init.xavier_normal_(self.indv_b)
        self.x_noisy_encoder = MLP([1] + 2 * [gene_init_dim])
        self.decoder = MLP(4 * [gene_init_dim])
        self.ln = nn.LayerNorm(gene_init_dim)
        self.gnn = MultiLayerGAT([gene_init_dim, gene_init_dim], [heads], edges)
        self.mole_encoder = MoleEncoder(mole_dim, gene_init_dim)
        self.gene_mole_encoder = MLP([2 * gene_init_dim] + 2 * [gene_init_dim])

    def forward(self, x_noisy, mole, dosage, cell_type_emb=None):
        b = x_noisy.shape[0]
        g = self.ln(self.gene_embedding.weight.expand(b, -1, -1))
        g = g + self.ln(self.x_noisy_encoder(x_noisy.unsqueeze(-1)))
        if cell_type_emb is not None:
            g = g + self.ln(self.cell_type_encoder(cell_type_emb)).unsqueeze(1)
        m = self.mole_encoder(mole, dosage)
        g = self.gene_mole_encoder(torch.cat([g, m[:, None, :].expand(-1, self.gene_num, -1)], dim=2))
        f = self.decoder(self.gnn(g))
        return (f * self.indv_w).sum(-1) + self.indv_b + x_noisy


class MaskModel(nn.Module):
    def __init__(self, gene_num, edges, gene_init_dim=64, mole_dim=512, heads=2, cell_type_num=1, cell_type_embed_dim=512):
        super().__init__()
        several = cell_type_num > 1
        self.cell_type_embedding = nn.Embedding(cell_type_num, cell_type_embed_dim) if several else None
        self.control_net = MaskNet(gene_num, edges, gene_init_dim, mole_dim, heads,
                                   cell_type_embed_dim=cell_type_embed_dim if several else None)

    def _cell(self, cell_type):
        if self.cell_type_embedding is None:
            return None
        if cell_type is None:
            raise ValueError("this mask model has several cell types: pass cell_type")
        return self.cell_type_embedding(cell_type.reshape(-1))

    def loss(self, x, mole, dosage, ctrl_mean, ctrl_var, cell_type=None):
        x_noisy = ctrl_mean + torch.sqrt(ctrl_var) * torch.randn_like(x)
        prob = torch.sigmoid(self.control_net(x_noisy, mole, dosage, self._cell(cell_type)))
        return F.binary_cross_entropy(prob, (x != 0).float())

    @torch.no_grad()
    def predict(self, x_ctrl, mole, dosage, cell_type=None):
        return torch.sigmoid(self.control_net(x_ctrl, mole, dosage, self._cell(cell_type)))
