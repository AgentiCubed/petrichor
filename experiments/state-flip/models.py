"""
The two competing valence heads.

Both are the SAME small MLP regressor with the SAME hidden architecture and
training budget. The only difference is the input:

  StateBlindBaseline   input = molecule descriptors           (D dims)
  StateConditionedHead input = molecule descriptors + state    (D + 1 dims)

That single extra scalar is the entire thesis under test. Holding everything
else fixed isolates the question to "does having a place to put internal state
let the model represent a valence flip the descriptor-only model cannot?"

Implementation note: we use sklearn's MLPRegressor (CPU, seconds to train, no
heavy deps) rather than torch. The architectural claim does not depend on the
optimiser; it depends on what is in the input. A bigger net would not rescue the
baseline, because the limit is informational (no state input), not capacity.
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HIDDEN_LAYERS = (128, 128)
MAX_ITER = 3000
SEED = 0


def _make_mlp():
    # StandardScaler in front: unscaled descriptors + a [0,1] state scalar train
    # poorly under adam; scaling lets both models actually converge. tol/no-change
    # are loosened so training does not stop short of the fit.
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=HIDDEN_LAYERS,
            activation="relu",
            solver="adam",
            alpha=1e-4,
            max_iter=MAX_ITER,
            tol=1e-6,
            n_iter_no_change=60,
            random_state=SEED,
            early_stopping=False,
        ),
    )


class StateBlindBaseline:
    """Descriptor-only valence model. The status quo: structure -> valence label.

    It has NO state input, so for a fixed molecule it emits one fixed number.
    By construction it cannot represent a molecule whose valence flips with state.
    """

    name = "state-blind baseline (descriptors only)"

    def __init__(self):
        self.mlp = _make_mlp()

    def fit(self, X_desc, state, y):
        self.mlp.fit(X_desc, y)          # state is deliberately ignored
        return self

    def predict(self, X_desc, state):
        return self.mlp.predict(X_desc)  # state ignored -> same output at any s


class StateConditionedHead:
    """Valence head conditioned on an internal-state scalar.

    structure x state -> valence. The state scalar is concatenated to the
    descriptor vector, giving the model somewhere to put the body's stake.
    """

    name = "state-conditioned head (descriptors + state)"

    def __init__(self):
        self.mlp = _make_mlp()

    def _augment(self, X_desc, state):
        return np.hstack([X_desc, np.asarray(state).reshape(-1, 1)])

    def fit(self, X_desc, state, y):
        self.mlp.fit(self._augment(X_desc, state), y)
        return self

    def predict(self, X_desc, state):
        return self.mlp.predict(self._augment(X_desc, state))
