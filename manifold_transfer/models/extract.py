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
