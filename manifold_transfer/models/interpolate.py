"""Walk DistilGPT2's weights back toward the GPT-2 layers it was initialised from.

DistilGPT2 did not start from scratch: the Hugging Face distillation recipe
(``examples/research_projects/distillation/scripts/extract.py``) copies GPT-2's
token and position embeddings and teacher blocks ``[0, 2, 4, 7, 9, 11]`` into the
six student blocks, and distillation moves them from there. So the straight line

    θ(t) = (1 − t) · θ_DistilGPT2  +  t · θ_GPT2[0, 2, 4, 7, 9, 11]

runs from the trained student (``t = 0``) back to its *pruned-teacher
initialisation* (``t = 1``), a model with the student's architecture whose every
weight is a teacher weight. Tracking the concept readouts along ``t`` asks a
question no post-hoc audit can: did distillation *build* the student's weekday
geometry, or *break* one the initialisation already had? Interpolating one block
at a time localises it to the block whose distillation update creates or
destroys the loop. Layer-wise interpolation of this kind is the setting of
Adilova et al. (*Layer-wise Linear Mode Connectivity*, arXiv:2307.06966);
Haskins & Adams (*Distilled Circuits*, arXiv:2505.10822) study this exact pair
and find the student compresses teacher components.

The parent-block map is an assumption about the released checkpoint (the
recipe's default); :func:`block_lineage` checks it from the weights before
anything is read along the path — each student block should be most similar to
its putative parent.
"""

from __future__ import annotations

import copy
import re
from typing import Iterable, Sequence

import numpy as np

DISTILGPT2_PARENT_BLOCKS: tuple[int, ...] = (0, 2, 4, 7, 9, 11)

_BLOCK = re.compile(r"^transformer\.h\.(\d+)\.(.*)$")


def _teacher_name(student_name: str, parents: Sequence[int]) -> str:
    m = _BLOCK.match(student_name)
    if m is None:
        return student_name
    return f"transformer.h.{parents[int(m.group(1))]}.{m.group(2)}"


def block_lineage(teacher, student) -> np.ndarray:
    """Cosine similarity between the flattened weights of every student block and
    every teacher block, ``(n_student_blocks, n_teacher_blocks)``. Row ``b``'s
    argmax is the teacher block student block ``b`` most resembles."""
    import torch

    def flat(model, idx):
        ps = [p.detach().float().reshape(-1) for n, p in model.named_parameters()
              if n.startswith(f"transformer.h.{idx}.")]
        return torch.cat(ps)

    n_s = len(student.transformer.h)
    n_t = len(teacher.transformer.h)
    t_flat = [flat(teacher, j) for j in range(n_t)]
    out = np.empty((n_s, n_t))
    for i in range(n_s):
        s = flat(student, i)
        for j in range(n_t):
            out[i, j] = float(torch.nn.functional.cosine_similarity(s, t_flat[j], dim=0))
    return out


def interpolate_student(
    teacher,
    student,
    t: float,
    *,
    parents: Sequence[int] = DISTILGPT2_PARENT_BLOCKS,
    blocks: Iterable[int] | None = None,
    include_embeddings: bool = True,
):
    """A copy of ``student`` with weights ``(1 − t) θ_student + t θ_parent``.

    ``blocks`` restricts the move to those student blocks (default: all);
    ``include_embeddings`` also moves ``wte``/``wpe``/``ln_f`` (tied ``lm_head``
    follows ``wte``). Buffers are left as they are.
    """
    import torch

    model = copy.deepcopy(student)
    t_params = dict(teacher.named_parameters())
    chosen = None if blocks is None else set(int(b) for b in blocks)
    with torch.no_grad():
        for name, p in model.named_parameters():
            m = _BLOCK.match(name)
            if m is not None:
                if chosen is not None and int(m.group(1)) not in chosen:
                    continue
            elif not include_embeddings:
                continue
            src = t_params.get(_teacher_name(name, parents))
            if src is None or src.shape != p.shape:
                continue
            p.mul_(1.0 - t).add_(src.to(p.dtype), alpha=t)
    return model.eval()


def lm_loss(model, tokenizer, texts: Sequence[str], *, device: str = "cpu", max_length: int = 128) -> float:
    """Mean next-token cross-entropy (nats/token) over ``texts`` — the barrier
    along the path."""
    import torch

    total, count = 0.0, 0
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
            if enc["input_ids"].shape[1] < 2:
                continue
            out = model(**enc, labels=enc["input_ids"])
            n = enc["input_ids"].shape[1] - 1
            total += float(out.loss) * n
            count += n
    return total / max(count, 1)
