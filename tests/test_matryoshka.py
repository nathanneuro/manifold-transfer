"""Matryoshka manifold supports: a planted circle must be found and dated."""

from __future__ import annotations

import numpy as np
import pytest

from manifold_transfer.matryoshka import (
    budget_grid,
    open_order_test,
    support_overlap,
    topology_onset,
)


def _planted_circle(n=7, d=24, carriers=(3, 11), noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    theta = 2 * np.pi * np.arange(n) / n
    x = rng.normal(scale=noise, size=(n, d))
    x[:, carriers[0]] += np.cos(theta)
    x[:, carriers[1]] += np.sin(theta)
    # distractor coordinates carry a large *unordered* signal
    x[:, [0, 1, 2]] += rng.normal(scale=3.0, size=(n, 3))
    return x


def test_onset_at_the_planted_carriers():
    x = _planted_circle()
    d = x.shape[1]
    # carriers first, quiet coordinates next, the loud unordered ones last
    good = np.r_[[3, 11], np.setdiff1d(np.arange(d), [0, 1, 2, 3, 11]), [0, 1, 2]]
    on = topology_onset(x, good, [2, 3, 6, 12, 21], topology="circle")
    assert on.onset_k == 2
    # ...and appending the loud inert coordinates buries the loop again
    assert topology_onset(x, good, [24]).p_values[0] > 1 / 360
    assert on.p_values[0] <= 1 / 360 + 1e-12
    # a ranking that leads with the loud unordered coordinates has no early onset
    bad = np.r_[[0, 1, 2], np.setdiff1d(np.arange(d), [0, 1, 2])]
    on_bad = topology_onset(x, bad, [2], topology="circle")
    assert on_bad.p_values[0] > 0.01


def test_open_order_test_finds_a_line():
    t = np.linspace(0, 1, 7)
    x = np.c_[t, t**2]
    res = open_order_test(x)
    assert res.exhaustive and res.p_value == pytest.approx(1 / res.n_null)
    rng = np.random.default_rng(1)
    assert open_order_test(rng.normal(size=(7, 5))).p_value > 0.01


def test_support_overlap_and_chance():
    d = 50
    a = np.arange(d)
    ov = support_overlap(a, a, [5, 10])
    assert np.allclose(ov.jaccard, 1.0)
    rev = support_overlap(a, a[::-1], [5, 25])
    assert rev.jaccard[0] == 0.0
    assert np.all(ov.chance < 1) and np.all(ov.chance > 0)


def test_budget_grid():
    g = budget_grid(768)
    assert g[0] == 2 and g[-1] == 768 and np.all(np.diff(g) > 0)


torch = pytest.importorskip("torch")


def test_sigmoid_topk_sums_to_k_and_learns_the_carriers():
    from manifold_transfer.models.matryoshka import (
        budget_curve,
        fisher_rao_torch,
        learn_scores,
        sigmoid_topk,
    )

    s = torch.randn(30)
    for k in (1, 7, 29):
        assert abs(float(sigmoid_topk(s, k).sum()) - k) < 1e-3

    # toy "model": the output distribution reads only coordinates 3 and 11 (a
    # circle); the rest are loud but behaviourally inert.
    x = torch.as_tensor(_planted_circle(n=12, carriers=(3, 11), seed=2), dtype=torch.float32)
    d = x.shape[1]
    readout = torch.zeros(d, 12)
    ang = 2 * np.pi * np.arange(12) / 12
    readout[3] = torch.as_tensor(4 * np.cos(ang), dtype=torch.float32)
    readout[11] = torch.as_tensor(4 * np.sin(ang), dtype=torch.float32)
    gen = np.random.default_rng(0)

    def logp(h):
        return torch.log_softmax(h @ readout, dim=-1)

    def loss_fn(mask):
        b = torch.as_tensor(gen.integers(0, 12, size=16))
        src = (b + torch.as_tensor(gen.integers(1, 12, size=16))) % 12
        h = mask * x[b] + (1 - mask) * x[src]
        return fisher_rao_torch(logp(h), logp(x[b])).mean()

    res = learn_scores(d, loss_fn, steps=300, lr=0.1, seed=0)
    assert set(res.ranking[:2].tolist()) == {3, 11}
    curve = budget_curve(loss_fn, res.ranking, [1, 2, d - 1])
    assert curve[1] < 0.2 * curve[0]


def test_weight_interpolation_on_tiny_gpt2():
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel

    from manifold_transfer.models.interpolate import block_lineage, interpolate_student

    torch.manual_seed(0)
    teacher = GPT2LMHeadModel(GPT2Config(n_layer=4, n_embd=16, n_head=2, vocab_size=50, n_positions=16))
    student = GPT2LMHeadModel(GPT2Config(n_layer=2, n_embd=16, n_head=2, vocab_size=50, n_positions=16))
    parents = (0, 3)
    # initialise the student from its parents, then perturb it (a "distillation")
    init = interpolate_student(teacher, student, 1.0, parents=parents)
    with torch.no_grad():
        for p in init.parameters():
            p.add_(0.01 * torch.randn_like(p))
    lin = block_lineage(teacher, init)
    assert list(lin.argmax(axis=1)) == [0, 3]
    back = interpolate_student(teacher, init, 1.0, parents=parents)
    tp = dict(teacher.named_parameters())
    for name, p in back.named_parameters():
        if name.startswith("transformer.h.1."):
            assert torch.allclose(p, tp[name.replace("h.1.", "h.3.")])
    half = interpolate_student(teacher, init, 0.5, parents=parents, blocks=[0], include_embeddings=False)
    hp = dict(half.named_parameters())
    ip = dict(init.named_parameters())
    assert torch.allclose(hp["transformer.h.1.attn.c_attn.weight"], ip["transformer.h.1.attn.c_attn.weight"])
    assert torch.allclose(hp["transformer.wte.weight"], ip["transformer.wte.weight"])


def test_concept_interchange_loss_runs_on_tiny_gpt2():
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    from manifold_transfer.models.matryoshka import concept_interchange_loss, learn_scores

    words = ["<eos>", "it", "is", "on", "mon", "tue", "wed", "thu"]
    tk = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<eos>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<eos>")
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_embd=16, n_head=2, vocab_size=len(words), n_positions=8))
    loss_fn, hidden = concept_interchange_loss(
        model, ["mon", "tue", "wed", "thu"], ["it is {}", "on {}"], layer=1, tokenizer=tok, batch_size=4
    )
    assert hidden == 16
    full = torch.ones(16)
    assert float(loss_fn(full)) < 1e-2  # nothing patched: output unchanged
    assert float(loss_fn(torch.zeros(16))) > 0  # everything patched from the source
    res = learn_scores(16, loss_fn, steps=10)
    assert res.scores.shape == (16,)
