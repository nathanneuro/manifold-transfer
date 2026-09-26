"""The concept manifold as a cartogram of the model's prior.

A diffusion cartogram (Gastner & Newman, *Diffusion-based method for producing
density-equalizing maps*, PNAS 101:7499, 2004) warps a map so that each region's
area is proportional to its population. Efficient coding says a population code
does the same to its stimulus axis: Fisher information is allocated so that
``√I(θ) ∝ p(θ)`` (Ganguli & Simoncelli, Neural Computation 26:2103, 2014), and
other loss functions give other exponents (Wang, Stocker & Lee, Neural
Computation 28:2656, 2016). Combined with the §2.1 speed law, the concept
manifold should be a cartogram of the prior:

    log spacing_i  =  γ · log p̄_i  +  c,        γ = 1 for infomax,

where ``p̄_i`` is the prior mass of the two items bounding step ``i``. Numbers and
years are where the prior is known and lumpy — number-word frequency falls like
``n^{-2}`` with spikes at 10, 12, 15, 20, 50, 100 (Dehaene & Mehler, Cognition
43:1, 1992), and year mentions spike at 1776, 1914, 1945, 2000. GPT-2 small's
years are already known to be isometric to ``log(2019 − year)`` (Modell,
Rubin-Delanchy & Whiteley, arXiv:2505.18235); if year frequency falls like
``1/(2019 − year)``, infomax gives exactly that.

The cross-model point: teacher and student share a training corpus, hence a
prior. If both draw the same cartogram, then "metric differs across models"
(Gröger et al.) narrows to "metric differs because priors differ" — for a
distill it should *transfer*.

Readouts: :func:`cartogram_fit` (γ with a mediation check — does the prior act
only through d_FR?), and :func:`landmark_bulge` (are the pre-registered
landmarks stretched beyond the smooth trend, against a null that moves the
landmarks?). The prior itself is read from the model (the mass it puts on each
item's token in neutral contexts) by the experiment script; a corpus count
vector can be passed instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


def pair_prior(prior: Any, topology: str = "interval") -> np.ndarray:
    """Geometric mean of the prior mass of the two items bounding each step."""
    p = np.asarray(prior, dtype=np.float64)
    if topology == "circle":
        return np.sqrt(p * np.roll(p, -1))
    return np.sqrt(p[:-1] * p[1:])


@dataclass
class CartogramFit:
    gamma: float  # slope of log spacing on log prior
    intercept: float
    r2: float
    gamma_given_fr: float | None  # prior slope with log d_FR in the model (mediation)
    fr_coef: float | None  # log d_FR slope with log prior in the model
    gamma_fr: float | None  # slope of log d_FR on log prior (behaviour-side cartogram)


def cartogram_fit(spacing: Any, prior_pairs: Any, d_fr: Any | None = None) -> CartogramFit:
    """Least squares on logs. With ``d_fr`` also fits the two-predictor model
    ``log s = a log p̄ + b log d_FR + c``: if the prior acts only through
    behaviour, ``a → 0`` and ``b → 1``."""
    ls = np.log(np.asarray(spacing, dtype=np.float64))
    lp = np.log(np.asarray(prior_pairs, dtype=np.float64))
    x = np.c_[lp, np.ones_like(lp)]
    coef, *_ = np.linalg.lstsq(x, ls, rcond=None)
    pred = x @ coef
    r2 = 1.0 - float(np.sum((ls - pred) ** 2) / np.sum((ls - ls.mean()) ** 2))
    g_given = fr_c = g_fr = None
    if d_fr is not None:
        lf = np.log(np.asarray(d_fr, dtype=np.float64))
        c2, *_ = np.linalg.lstsq(np.c_[lp, lf, np.ones_like(lp)], ls, rcond=None)
        g_given, fr_c = float(c2[0]), float(c2[1])
        g_fr = float(np.linalg.lstsq(x, lf, rcond=None)[0][0])
    return CartogramFit(float(coef[0]), float(coef[1]), r2, g_given, fr_c, g_fr)


@dataclass
class LandmarkBulge:
    mean_residual: float  # mean detrended log-spacing at the landmark steps
    p_value: float  # fraction of shifted landmark sets with at least that mean
    n_null: int


def detrend(log_values: Any, *, window: int = 15) -> np.ndarray:
    """Subtract a centred running median (a smooth trend that a single spike
    cannot drag)."""
    v = np.asarray(log_values, dtype=np.float64)
    half = window // 2
    trend = np.array([np.median(v[max(0, i - half): i + half + 1]) for i in range(v.size)])
    return v - trend


def landmark_bulge(
    log_spacing: Any,
    landmark_steps: Sequence[int],
    *,
    window: int = 15,
    n_null: int = 5000,
    seed: int = 0,
) -> LandmarkBulge:
    """Are the steps adjacent to pre-registered landmarks stretched beyond trend?
    The null draws random step sets of the same size, avoiding the ends."""
    r = detrend(log_spacing, window=window)
    idx = np.asarray(landmark_steps, dtype=int)
    obs = float(r[idx].mean())
    rng = np.random.default_rng(seed)
    pool = np.arange(window // 2, r.size - window // 2)
    null = np.array([r[rng.choice(pool, size=idx.size, replace=False)].mean() for _ in range(n_null)])
    return LandmarkBulge(obs, float((1 + np.sum(null >= obs)) / (n_null + 1)), n_null)
