"""
2D chemical plume arena — the cheapest world where valence must drive action.

An agent moves on a grid. Multiple odor sources emit Gaussian concentration
fields. At each cell the agent "smells" a concentration-weighted mixture of
source descriptors, carries an internal satiety scalar, and may FORAGE
(consume at the current cell).

Reward is homeostatic / stake-shaped, not label-shaped:
  - foraging a FOOD plume while hungry  -> positive reward, satiety rises
  - foraging a FOOD plume while sated   -> negative reward (the flip)
  - foraging a TOXIN plume              -> always negative (damage)
  - foraging NEUTRAL / empty air        -> near zero

This is the behavioural half of the thesis: a live approach/avoid signal tied
to something the system has reason to care about (its own satiety / integrity).

No external physics engine. Pure numpy. CPU-only. Reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Actions
UP, DOWN, LEFT, RIGHT, STAY, FORAGE = 0, 1, 2, 3, 4, 5
ACTION_NAMES = ["up", "down", "left", "right", "stay", "forage"]
N_ACTIONS = 6


@dataclass
class SourceSpec:
    """One odor source in the arena."""

    name: str
    kind: str  # "food" | "toxin" | "neutral"
    position: Tuple[int, int]
    descriptors: np.ndarray  # (D,)
    base_valence: float
    strength: float = 1.0
    sigma: float = 3.5


class PlumeArena:
    """Discrete plume world with internal satiety state."""

    def __init__(
        self,
        width: int = 15,
        height: int = 15,
        sources: Optional[List[SourceSpec]] = None,
        d_descriptors: int = 16,
        amplitude: float = 1.5,
        hunger_rate: float = 0.02,
        max_steps: int = 80,
        seed: int = 0,
    ):
        self.width = width
        self.height = height
        self.d = d_descriptors
        self.amplitude = amplitude
        self.hunger_rate = hunger_rate
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.sources = sources if sources is not None else self._default_sources()
        self._fields = self._build_fields()
        self.reset(seed=seed)

    def _default_sources(self) -> List[SourceSpec]:
        rng = self.rng
        D = self.d

        def desc(bias: float):
            v = rng.standard_normal(D)
            v = v / (np.linalg.norm(v) + 1e-8)
            v[0] += bias  # make kinds linearly separable-ish
            return v

        # Place sources in corners / edges so navigation matters.
        w, h = self.width, self.height
        return [
            SourceSpec("food_a", "food", (1, 1), desc(2.0), base_valence=0.1, strength=1.2, sigma=3.0),
            SourceSpec("food_b", "food", (w - 2, h - 2), desc(1.8), base_valence=0.05, strength=1.0, sigma=3.0),
            SourceSpec("toxin", "toxin", (1, h - 2), desc(-2.0), base_valence=-0.8, strength=1.1, sigma=3.0),
            SourceSpec("neutral", "neutral", (w - 2, 1), desc(0.0), base_valence=0.0, strength=0.7, sigma=3.5),
        ]

    def _build_fields(self) -> Dict[str, np.ndarray]:
        """Precompute concentration field per source: shape (H, W)."""
        yy, xx = np.mgrid[0:self.height, 0:self.width]
        fields = {}
        for src in self.sources:
            sy, sx = src.position
            dist2 = (yy - sy) ** 2 + (xx - sx) ** 2
            fields[src.name] = src.strength * np.exp(-dist2 / (2.0 * src.sigma ** 2))
        return fields

    def reset(self, seed: Optional[int] = None, satiety: Optional[float] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # Start near center, not on a source.
        self.y = self.height // 2
        self.x = self.width // 2
        self.satiety = float(satiety) if satiety is not None else float(self.rng.uniform(0.0, 1.0))
        self.steps = 0
        self.total_reward = 0.0
        self.done = False
        self.last_info: dict = {}
        return self.observe()

    def concentration_vector(self, y: Optional[int] = None, x: Optional[int] = None) -> np.ndarray:
        """Per-source concentration at a cell."""
        y = self.y if y is None else y
        x = self.x if x is None else x
        return np.array([self._fields[s.name][y, x] for s in self.sources], dtype=float)

    def smell_descriptors(self, y: Optional[int] = None, x: Optional[int] = None) -> np.ndarray:
        """Concentration-weighted mixture of source descriptors (what the nose gets)."""
        c = self.concentration_vector(y, x)
        total = c.sum()
        if total < 1e-8:
            return np.zeros(self.d, dtype=float)
        weights = c / total
        desc = np.zeros(self.d, dtype=float)
        for w, src in zip(weights, self.sources):
            desc += w * src.descriptors
        # Scale by log1p(total) so intensity is felt, not just identity.
        return desc * np.log1p(total)

    def true_valence_at(self, y: Optional[int] = None, x: Optional[int] = None,
                        satiety: Optional[float] = None) -> float:
        """Ground-truth stake at a cell given current (or supplied) satiety."""
        s = self.satiety if satiety is None else satiety
        c = self.concentration_vector(y, x)
        total = 0.0
        for conc, src in zip(c, self.sources):
            if src.kind == "food":
                # satiety flip: +A when starving, -A when sated
                v = src.base_valence + self.amplitude * (1.0 - 2.0 * s)
            elif src.kind == "toxin":
                v = src.base_valence - self.amplitude  # always bad
            else:
                v = src.base_valence
            total += conc * v
        return float(total)

    def observe(self) -> dict:
        """Observation dict the agent may use."""
        return {
            "descriptors": self.smell_descriptors(),
            "satiety": np.array([self.satiety], dtype=float),
            "position": (self.y, self.x),
            "concentrations": self.concentration_vector(),
            "true_valence": self.true_valence_at(),
        }

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Episode finished; call reset().")

        reward = 0.0
        info = {"foraged": False, "forage_kind": None}

        if action == UP:
            self.y = max(0, self.y - 1)
        elif action == DOWN:
            self.y = min(self.height - 1, self.y + 1)
        elif action == LEFT:
            self.x = max(0, self.x - 1)
        elif action == RIGHT:
            self.x = min(self.width - 1, self.x + 1)
        elif action == STAY:
            pass
        elif action == FORAGE:
            reward, kind = self._forage()
            info["foraged"] = True
            info["forage_kind"] = kind
        else:
            raise ValueError(f"Invalid action {action}")

        # Slow hunger drift toward starving if not eating.
        if action != FORAGE:
            self.satiety = float(np.clip(self.satiety - self.hunger_rate, 0.0, 1.0))

        # Small step cost keeps agents from dithering forever.
        reward -= 0.01

        self.steps += 1
        self.total_reward += reward
        self.done = self.steps >= self.max_steps
        self.last_info = info
        obs = self.observe()
        return obs, reward, self.done, info

    def _forage(self) -> Tuple[float, Optional[str]]:
        """Consume at current cell. Reward = stake, not label."""
        c = self.concentration_vector()
        if c.sum() < 0.05:
            return -0.05, None  # foraging empty air

        # Dominant source by concentration decides the event.
        idx = int(np.argmax(c))
        src = self.sources[idx]
        intensity = float(c[idx])

        if src.kind == "food":
            # The flip, lived: delicious hungry, nauseating full.
            local_v = src.base_valence + self.amplitude * (1.0 - 2.0 * self.satiety)
            reward = intensity * local_v
            # Eating raises satiety proportional to intensity.
            self.satiety = float(np.clip(self.satiety + 0.25 * intensity, 0.0, 1.0))
            return float(reward), "food"
        if src.kind == "toxin":
            reward = intensity * (src.base_valence - self.amplitude)
            # Toxin does not feed you.
            return float(reward), "toxin"
        # neutral
        return float(intensity * src.base_valence * 0.1), "neutral"

    def valence_map(self, satiety: Optional[float] = None) -> np.ndarray:
        """Full-grid true valence at a given satiety — for plots / oracles."""
        s = self.satiety if satiety is None else satiety
        grid = np.zeros((self.height, self.width), dtype=float)
        for y in range(self.height):
            for x in range(self.width):
                grid[y, x] = self.true_valence_at(y, x, satiety=s)
        return grid
