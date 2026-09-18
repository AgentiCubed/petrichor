"""
Petrichor — grounded valence, with olfaction as its cleanest vehicle.

A model whose output for a stimulus is not a label but a stake:
good-for-me / bad-for-me, tied to something the system has reason to care about.
"""

__version__ = "0.3.0"

from petrichor.data import make_dataset, load_dream, split_by_molecule
from petrichor.models import (
    StateBlindBaseline,
    StateConditionedHead,
    ReceptorRoutedHead,
    StateAsFeatureWrongObjective,
)
from petrichor.metrics import (
    irreducible_baseline_food_mse,
    evaluate_flip,
    score_choices,
)

__all__ = [
    "__version__",
    "make_dataset",
    "load_dream",
    "split_by_molecule",
    "StateBlindBaseline",
    "StateConditionedHead",
    "ReceptorRoutedHead",
    "StateAsFeatureWrongObjective",
    "irreducible_baseline_food_mse",
    "evaluate_flip",
    "score_choices",
]
