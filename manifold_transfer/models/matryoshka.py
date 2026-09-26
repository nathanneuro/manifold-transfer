"""Matryoshka attribution of a concept manifold to its supporting components.

Port of the core of MAttr (Arora et al., *Matryoshka Attribution: Learning to
Attribute Language Model Outputs to Representations and Weights*,
arXiv:2609.25518; reference code github.com/aryamanarora/matryoshka-attribution)
to the manifold setting. MAttr learns one score per internal variable such that,
for *every* budget ``k``, keeping the top-``k`` variables on the base run and
patching the rest from a source run preserves the base run's output. The budget
is resampled each step, so one training run yields a nested ranking rather than
a circuit at one sparsity.

The manifold reading. Take base and source prompts that differ only in which
item of a concept they name ("It is Monday" vs "It is Thursday"). Patching the
non-kept coordinates of the residual stream from the source moves the base run
*along the concept manifold*. The loss is the Fisher-Rao distance between the
base run's next-token distribution and the patched one — the same metric the
§2.1 speed law is written in — so the learned ranking orders coordinates by how
much of the concept's *behavioural* position they carry. The nested supports
then feed the pure-numpy analysis in :mod:`manifold_transfer.matryoshka`: at
what budget does the loop appear, does the distill need a larger fraction of
its width, do two concepts share a support.

Two pieces, both torch:

- :func:`sigmoid_topk` — the differentiable top-k mask of Wijk et al. (2025),
  ``sigma((s - tau) / T)`` with ``tau`` found by bisection so the mask sums to
  ``k``, and the implicit-differentiation backward (as in the reference code).
- :func:`learn_scores` — the MAttr loop (uniform ``k``, Adam), taking an
  environment ``loss_fn(mask) -> loss`` exactly like the reference trainer.

:func:`concept_interchange_loss` builds that environment for a Hugging Face
causal LM over the residual stream at one layer, at the last token.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

try:  # the models extra
    import torch
    from torch.autograd import Function
except ImportError:  # pragma: no cover - exercised only without torch
    torch = None  # type: ignore[assignment]
    Function = object  # type: ignore[assignment,misc]

_EPS = 1e-8


class _SigmoidTopK(Function):  # type: ignore[misc]
    @staticmethod
    def forward(ctx, scores, k, T, n_iters):  # noqa: N803 - T is the paper's name
        lo = scores.min(dim=-1, keepdim=True).values - 10 * T
        hi = scores.max(dim=-1, keepdim=True).values + 10 * T
        with torch.no_grad():
            for _ in range(n_iters):
                mid = (lo + hi) / 2
                f_mid = torch.special.expit((scores - mid) / T).sum(dim=-1, keepdim=True)
                lo = torch.where(f_mid > k, mid, lo)
                hi = torch.where(f_mid > k, hi, mid)
        tau = (lo + hi) / 2
        mask = torch.special.expit((scores - tau) / T)
        ctx.save_for_backward(mask)
        ctx.T = T
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        # Implicit differentiation of sum_i sigma((s_i - tau)/T) = k:
        #   dL/ds_j = (sp_j / T) * (g_j - sum_i g_i sp_i / sum_i sp_i),  sp = m(1-m)
        (mask,) = ctx.saved_tensors
        sp = mask * (1.0 - mask)
        sp_sum = sp.sum(dim=-1, keepdim=True).clamp(min=_EPS)
        gsp = (grad_output * sp).sum(dim=-1, keepdim=True)
        return (sp / ctx.T) * (grad_output - gsp / sp_sum), None, None, None


def sigmoid_topk(scores, k: float, T: float = 0.5, n_iters: int = 50):  # noqa: N803
    """Soft top-``k`` mask over the last axis of ``scores``: entries in ``(0, 1)``
    summing to ``k``, differentiable in ``scores``."""
    if torch is None:
        raise ImportError("sigmoid_topk needs torch (the 'models' extra)")
    return _SigmoidTopK.apply(scores, float(k), float(T), int(n_iters))


@dataclass
class MatryoshkaResult:
    scores: np.ndarray  # one score per variable; argsort descending is the nested ranking
    loss_log: list[float] = field(default_factory=list)
    k_log: list[float] = field(default_factory=list)

    @property
    def ranking(self) -> np.ndarray:
        return np.argsort(-self.scores, kind="stable")


def learn_scores(
    total: int,
    loss_fn: Callable[["torch.Tensor"], "torch.Tensor"],
    *,
    steps: int = 500,
    T: float = 0.5,  # noqa: N803
    lr: float = 0.05,
    seed: int = 0,
    device: str = "cpu",
) -> MatryoshkaResult:
    """The MAttr loop: scores start at zero; each step draws ``k ~ U{1..total-1}``,
    builds ``sigmoid_topk(scores, k)`` and descends ``loss_fn(mask)``. In
    expectation this minimises the area under the loss-vs-budget curve, which is
    what makes the resulting ranking nested."""
    if torch is None:
        raise ImportError("learn_scores needs torch (the 'models' extra)")
    if total < 2:
        raise ValueError("need at least 2 variables to rank")
    gen = np.random.default_rng(seed)
    scores = torch.nn.Parameter(torch.zeros(total, device=device))
    opt = torch.optim.Adam([scores], lr=lr)
    out = MatryoshkaResult(scores=np.zeros(total))
    for _ in range(steps):
        k = int(gen.integers(1, total))
        opt.zero_grad()
        loss = loss_fn(sigmoid_topk(scores, k, T))
        loss.backward()
        opt.step()
        out.loss_log.append(float(loss.detach()))
        out.k_log.append(float(k))
    out.scores = scores.detach().cpu().numpy().astype(np.float64)
    return out


def budget_curve(
    loss_fn: Callable[["torch.Tensor"], "torch.Tensor"],
    ranking: Sequence[int],
    ks: Sequence[int],
    *,
    n_batches: int = 8,
    device: str = "cpu",
) -> np.ndarray:
    """Loss under the *hard* top-``k`` mask of ``ranking`` for each budget, averaged
    over ``n_batches`` calls of the environment: the behavioural half of the onset
    comparison (where does the patched output stop moving?)."""
    rank = np.asarray(ranking, dtype=int)
    out = np.empty(len(ks))
    with torch.no_grad():
        for i, k in enumerate(ks):
            mask = torch.zeros(rank.size, device=device)
            mask[torch.as_tensor(rank[: int(k)], device=device)] = 1.0
            out[i] = float(np.mean([float(loss_fn(mask)) for _ in range(n_batches)]))
    return out


def fisher_rao_torch(logp: "torch.Tensor", logq: "torch.Tensor") -> "torch.Tensor":
    """Row-wise Fisher-Rao distance ``2 arccos BC`` between two batches of
    log-distributions, clamped away from the arccos singularity so it can be
    back-propagated when the two rows coincide."""
    bc = torch.exp(0.5 * (logp + logq)).sum(dim=-1)
    return 2.0 * torch.arccos(bc.clamp(max=1.0 - 1e-6))


def concept_interchange_loss(
    model_name: str,
    items: Sequence[str],
    templates: Sequence[str],
    *,
    layer: int,
    batch_size: int = 16,
    seed: int = 0,
    device: str | None = None,
) -> tuple[Callable[["torch.Tensor"], "torch.Tensor"], int]:
    """Environment for :func:`learn_scores` on one concept of a causal LM.

    Variables are the residual-stream coordinates at the output of hidden-state
    index ``layer`` (so ``layer`` means the same as in
    :func:`manifold_transfer.models.extract.resolve_layer`; ``layer >= 1``), at
    the last token. Each call draws ``batch_size`` (template, base item, source
    item) triples with base != source, runs the base prompt with the kept
    coordinates on the base value and the rest patched to the source value
    (``h* = m h_base + (1 - m) h_source``, Definition 1 of MAttr), and returns
    the mean Fisher-Rao distance of the patched next-token distribution from the
    unpatched base one. Returns ``(loss_fn, hidden_size)``.

    Only the last token is patched, so every item must be one token under the
    template or the patch lands on a partial word; the caller's concept lists
    are chosen that way (and the base/source prompts share their prefix).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if layer < 1:
        raise ValueError("layer indexes hidden_states; patch a block output (layer >= 1)")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    blocks = _blocks(model)
    block = blocks[layer - 1]

    texts = [t.format(it) for t in templates for it in items]  # template-major here
    enc = tok(texts, return_tensors="pt", padding=True).to(device)
    last = enc["attention_mask"].sum(dim=1) - 1
    n_i = len(items)

    # Cache every prompt's hidden state at the patch site and its clean log-probs.
    captured: dict[str, torch.Tensor] = {}

    def grab(_m, _inp, output):
        captured["h"] = output[0] if isinstance(output, tuple) else output

    handle = block.register_forward_hook(grab)
    with torch.no_grad():
        logits = model(**enc).logits
    handle.remove()
    rows = torch.arange(len(texts), device=device)
    h_clean = captured["h"][rows, last].detach()  # (n_texts, hidden)
    logp_clean = torch.log_softmax(logits[rows, last].float(), dim=-1).detach()
    hidden = int(h_clean.shape[1])

    gen = np.random.default_rng(seed)
    state: dict[str, torch.Tensor] = {}

    def patch(_m, _inp, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h.clone()
        b_idx, s_idx, mask = state["b"], state["s"], state["mask"]
        pos = last[b_idx]
        r = torch.arange(h.shape[0], device=h.device)
        h[r, pos] = mask * h_clean[b_idx] + (1.0 - mask) * h_clean[s_idx]
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    def loss_fn(mask: torch.Tensor) -> torch.Tensor:
        t = gen.integers(0, len(templates), size=batch_size)
        bi = gen.integers(0, n_i, size=batch_size)
        si = (bi + gen.integers(1, n_i, size=batch_size)) % n_i  # source != base
        b_idx = torch.as_tensor(t * n_i + bi, device=device)
        s_idx = torch.as_tensor(t * n_i + si, device=device)
        state.update(b=b_idx, s=s_idx, mask=mask)
        h = block.register_forward_hook(patch)
        try:
            out = model(
                input_ids=enc["input_ids"][b_idx],
                attention_mask=enc["attention_mask"][b_idx],
            ).logits
        finally:
            h.remove()
        r = torch.arange(batch_size, device=device)
        logp = torch.log_softmax(out[r, last[b_idx]].float(), dim=-1)
        return fisher_rao_torch(logp, logp_clean[b_idx]).mean()

    return loss_fn, hidden


def _blocks(model):
    for path in ("transformer.h", "model.layers", "gpt_neox.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise ValueError("could not locate the transformer block list on this model")
