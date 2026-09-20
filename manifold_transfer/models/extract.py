"""White-box activation extraction from a local Hugging Face model.

Thin glue over `transformers`: run text through a model with
``output_hidden_states`` and capture the chosen layer's hidden state at the last
real (non-pad) token of each input. This is the model-touching front end that
turns concepts into the activation point clouds the discovery / chart / audit
layers consume. Requires the ``models`` extra (``torch``, ``transformers``).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def extract_last_token_activations(
    model_name: str,
    texts: Sequence[str],
    *,
    layer: int = -1,
    device: str | None = None,
    batch_size: int = 32,
    max_length: int = 64,
) -> np.ndarray:
    """Hidden state at ``layer`` for the last real token of each text.

    ``layer`` indexes ``output_hidden_states`` (0 = embeddings, -1 = final), so it
    is comparable across models of different depth. Returns ``(len(texts),
    hidden)``. Pads on the right and uses the attention mask to find each row's
    last real token, so padding never leaks into the captured vector.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = (
        AutoModel.from_pretrained(model_name, output_hidden_states=True)
        .to(device)
        .eval()
    )

    out: list[np.ndarray] = []
    texts = list(texts)
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        with torch.no_grad():
            result = model(**enc)
        hidden = result.hidden_states[layer]  # (B, T, H)
        last = enc["attention_mask"].sum(dim=1) - 1  # (B,) last real token index
        rows = torch.arange(hidden.size(0), device=hidden.device)
        vecs = hidden[rows, last]  # (B, H)
        out.append(vecs.float().cpu().numpy())

    return np.concatenate(out, axis=0)


def resolve_layer(model_name: str, depth_fraction: float) -> int:
    """Index into ``output_hidden_states`` at a relative depth, so the same
    fraction names comparable layers in models of different depth (0.5 → block 6
    of GPT-2's 12, block 3 of DistilGPT2's 6). Pre-register the fractions you
    read; sweeping 12 × 6 layer pairs and reporting the best is the
    look-elsewhere effect the notes warn about."""
    from transformers import AutoConfig

    if not 0.0 <= depth_fraction <= 1.0:
        raise ValueError("depth_fraction must be in [0, 1]")
    n_layers = int(AutoConfig.from_pretrained(model_name).num_hidden_layers)
    return int(round(depth_fraction * n_layers))


def extract_last_token_activations_and_distributions(
    model_name: str,
    texts: Sequence[str],
    *,
    layers: Sequence[int] = (-1,),
    device: str | None = None,
    batch_size: int = 32,
    max_length: int = 64,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Hidden states at each of ``layers`` **and** the next-token distribution,
    both at the last real token of each text, from one forward pass of the
    causal-LM head.

    Returns ``({layer: (n, hidden)}, probs)`` with ``probs`` ``(n, vocab)`` in
    float32. The distributions are the Fisher-law predictor's raw material
    (``manifold_transfer.fisher.behavioral_spacing``); the activations are the
    spacing it predicts. Read both from the same pass so they are matched.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = (
        AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
        .to(device)
        .eval()
    )

    acts: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    probs: list[np.ndarray] = []
    texts = list(texts)
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        with torch.no_grad():
            result = model(**enc)
        last = enc["attention_mask"].sum(dim=1) - 1
        rows = torch.arange(last.size(0), device=last.device)
        for layer in layers:
            hidden = result.hidden_states[layer]
            acts[layer].append(hidden[rows, last].float().cpu().numpy())
        logits = result.logits[rows, last].float()
        probs.append(torch.softmax(logits, dim=-1).cpu().numpy())

    return (
        {layer: np.concatenate(chunks, axis=0) for layer, chunks in acts.items()},
        np.concatenate(probs, axis=0),
    )
