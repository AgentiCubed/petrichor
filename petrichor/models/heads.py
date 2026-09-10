"""
Competing valence heads.

All share the same MLP capacity and training budget unless noted. Differences
are *informational / architectural*, which is the point:

  StateBlindBaseline          input = descriptors only
  StateConditionedHead        input = descriptors + state          (Lane B)
  ReceptorRoutedHead          descriptors -> receptor layer -> valence x state
                              (Lane C — thesis claim in the architecture)
  StateAsFeatureWrongObjective
                              input = descriptors + state, but trained on the
                              state-averaged target E_s[y|m]. Same sensors,
                              wrong objective — the same-info control for the
                              falsifiable core (03 §5.2).
"""

from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HIDDEN_LAYERS = (128, 128)
MAX_ITER = 3000
SEED = 0


def _make_mlp(seed: int = SEED, hidden=HIDDEN_LAYERS):
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            alpha=1e-4,
            max_iter=MAX_ITER,
            tol=1e-6,
            n_iter_no_change=60,
            random_state=seed,
            early_stopping=False,
        ),
    )


class StateBlindBaseline:
    """Descriptor-only valence. Structure -> one number. Cannot flip with state."""

    name = "state-blind baseline (descriptors only)"

    def __init__(self, seed: int = SEED):
        self.mlp = _make_mlp(seed=seed)

    def fit(self, X_desc, state, y):
        self.mlp.fit(X_desc, y)  # state deliberately ignored
        return self

    def predict(self, X_desc, state):
        return self.mlp.predict(X_desc)


class StateConditionedHead:
    """Lane B: structure x state -> valence."""

    name = "state-conditioned head (descriptors + state)"

    def __init__(self, seed: int = SEED):
        self.mlp = _make_mlp(seed=seed)

    def _augment(self, X_desc, state):
        return np.hstack([X_desc, np.asarray(state, dtype=float).reshape(-1, 1)])

    def fit(self, X_desc, state, y):
        self.mlp.fit(self._augment(X_desc, state), y)
        return self

    def predict(self, X_desc, state):
        return self.mlp.predict(self._augment(X_desc, state))


class StateAsFeatureWrongObjective:
    """Same-information control (falsifiable core).

    Sees state in the input (same sensors as the conditioned head) but is
    supervised on the *state-averaged* target per molecule:

        y_wrong(m) = E_s[y | m]   ≈ base(m)

    Optimal behaviour under that objective is to *ignore* state. Having the
    bits is not enough; the learning signal must be stake-shaped. This is the
    clean control against "just concatenate state and a bigger net will figure
    it out from the features alone."
    """

    name = "same-info control (state in input, state-averaged target)"

    def __init__(self, seed: int = SEED):
        self.mlp = _make_mlp(seed=seed)

    def _augment(self, X_desc, state):
        return np.hstack([X_desc, np.asarray(state, dtype=float).reshape(-1, 1)])

    def fit(self, X_desc, state, y, mol_id=None):
        y = np.asarray(y, dtype=float)
        if mol_id is None:
            # Fallback: empirical mean over the batch (weaker control).
            y_target = np.full_like(y, y.mean())
        else:
            mol_id = np.asarray(mol_id)
            y_target = np.empty_like(y)
            for m in np.unique(mol_id):
                mask = mol_id == m
                y_target[mask] = y[mask].mean()
        self.mlp.fit(self._augment(X_desc, state), y_target)
        return self

    def predict(self, X_desc, state):
        return self.mlp.predict(self._augment(X_desc, state))


class ReceptorRoutedHead:
    """Lane C: route valence through a receptor-binding layer.

    Architecture (thesis claim in the wiring, not just the input vector):

        descriptors  --(binding)-->  receptor activations r(m) in R^K
        valence      =  f( r(m), state )

    Binding is a fixed random projection into K "receptor" units with a soft
    non-linearity (softplus), standing in for affinity docking. The valence
    head only ever sees receptor population activity + state — never raw
    descriptors. Contact-shaped intermediate representation.

    K defaults to 64. A state-blind ablation (state ignored after receptors)
    is available via ``state_blind=True`` for controls.
    """

    name = "receptor-routed head (Lane C)"

    def __init__(
        self,
        n_receptors: int = 64,
        seed: int = SEED,
        state_blind: bool = False,
    ):
        self.n_receptors = n_receptors
        self.seed = seed
        self.state_blind = state_blind
        self.mlp = _make_mlp(seed=seed)
        self._W = None  # (D, K) binding matrix
        self._b = None  # (K,) receptor baselines

    def _init_receptors(self, d: int):
        rng = np.random.default_rng(self.seed + 17)
        # Unit-ish columns: each receptor has a preferred direction in desc space.
        W = rng.standard_normal((d, self.n_receptors)) / np.sqrt(d)
        b = rng.normal(0, 0.1, size=self.n_receptors)
        self._W = W
        self._b = b

    def bind(self, X_desc: np.ndarray) -> np.ndarray:
        """Receptor population activity for a batch of descriptor vectors."""
        X_desc = np.asarray(X_desc, dtype=float)
        if self._W is None:
            self._init_receptors(X_desc.shape[1])
        # softplus affinity: log(1 + exp(x)) — smooth, non-negative "docking"
        act = X_desc @ self._W + self._b
        return np.log1p(np.exp(-np.abs(act))) + np.maximum(act, 0)  # stable softplus

    def _augment(self, X_desc, state):
        r = self.bind(X_desc)
        if self.state_blind:
            return r
        return np.hstack([r, np.asarray(state, dtype=float).reshape(-1, 1)])

    def fit(self, X_desc, state, y):
        X_desc = np.asarray(X_desc, dtype=float)
        if self._W is None:
            self._init_receptors(X_desc.shape[1])
        self.mlp.fit(self._augment(X_desc, state), y)
        return self

    def predict(self, X_desc, state):
        return self.mlp.predict(self._augment(X_desc, state))

    def receptor_activity(self, X_desc: np.ndarray) -> np.ndarray:
        """Public accessor for representation probes."""
        return self.bind(X_desc)
