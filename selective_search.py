"""Selective search: a gate decides at each step whether to call the planner.

The agent only needs a planner with ``search(observation, budget)`` returning an
object with ``action`` and ``model_transitions`` (e.g. Ella's ``MCTSPlanner``)
and a PPO adapter ``policy(observation) -> (probabilities, value)``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli

PolicyFn = Callable[[Any], tuple[Sequence[float], float]]
FEATURE_NAMES = ("entropy", "max_prob", "margin", "value")


class Planner(Protocol):
    def search(self, observation: Any, budget: Any) -> Any: ...


def policy_features(probabilities: Sequence[float], value: float) -> np.ndarray:
    """Summarize how unsure PPO is: normalized entropy, top prob, top-2 margin, critic value."""
    p = np.asarray(probabilities, dtype=np.float32)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("expected a non-empty 1-D probability vector")
    if p.size == 1:
        return np.array([0.0, 1.0, 1.0, value], dtype=np.float32)
    entropy = float(-(p * np.log(np.clip(p, 1e-8, 1.0))).sum() / math.log(p.size))
    top2 = np.sort(p)[-2:]
    return np.array([entropy, top2[1], top2[1] - top2[0], value], dtype=np.float32)


@dataclass(frozen=True)
class GateDecision:
    search: bool
    prob: float  # P(search | features)
    log_prob: float  # log-prob of the decision taken (0 for fixed gates)
    value: float  # gate critic estimate (0 for fixed gates)


class Gate:
    trainable = False

    def decide(self, features: np.ndarray, *, explore: bool = False) -> GateDecision:
        raise NotImplementedError


class AlwaysGate(Gate):
    def decide(self, features, *, explore=False):
        return GateDecision(True, 1.0, 0.0, 0.0)


class NeverGate(Gate):
    def decide(self, features, *, explore=False):
        return GateDecision(False, 0.0, 0.0, 0.0)


class EntropyGate(Gate):
    """Heuristic baseline: search when PPO's normalized entropy is above a threshold."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def decide(self, features, *, explore=False):
        search = bool(features[0] > self.threshold)
        return GateDecision(search, float(search), 0.0, 0.0)


class LearnedGate(Gate, nn.Module):
    """Small actor-critic MLP: P(search) head and a value head, trained with PPO."""

    trainable = True

    def __init__(self, feature_dim: int = len(FEATURE_NAMES), hidden: int = 64, init_search_prob: float = 0.5):
        nn.Module.__init__(self)
        self.trunk = nn.Sequential(nn.Linear(feature_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.logit = nn.Linear(hidden, 1)
        self.value = nn.Linear(hidden, 1)
        nn.init.zeros_(self.logit.weight)
        nn.init.constant_(self.logit.bias, math.log(init_search_prob / (1 - init_search_prob)))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(features)
        return self.logit(h).squeeze(-1), self.value(h).squeeze(-1)

    @torch.no_grad()
    def decide(self, features, *, explore=False):
        logit, value = self(torch.as_tensor(features, dtype=torch.float32))
        dist = Bernoulli(logits=logit)
        search = dist.sample() if explore else (logit > 0).float()
        return GateDecision(bool(search), float(dist.probs), float(dist.log_prob(search)), float(value))

    def evaluate(self, features: torch.Tensor, searched: torch.Tensor):
        logit, value = self(features)
        dist = Bernoulli(logits=logit)
        return dist.log_prob(searched), dist.entropy(), value


@dataclass(frozen=True)
class StepResult:
    action: int
    searched: bool
    features: np.ndarray
    gate: GateDecision
    model_transitions: int
    elapsed_seconds: float
    search: Any | None  # the planner's result when searched


class SelectiveSearchAgent:
    def __init__(
        self,
        planner: Planner,
        budget: Any,
        policy: PolicyFn,
        gate: Gate,
        *,
        extra_features: Callable[[Any], Sequence[float]] | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.planner = planner
        self.budget = budget
        self.policy = policy
        self.gate = gate
        self.extra_features = extra_features
        self.rng = rng or np.random.default_rng()

    def features(self, observation: Any) -> tuple[np.ndarray, Sequence[float]]:
        probabilities, value = self.policy(observation)
        features = policy_features(probabilities, value)
        if self.extra_features is not None:
            features = np.concatenate([features, np.asarray(self.extra_features(observation), dtype=np.float32)])
        return features, probabilities

    def act(self, observation: Any, *, explore_gate: bool = False, sample_action: bool = False) -> StepResult:
        start = perf_counter()
        features, probabilities = self.features(observation)
        decision = self.gate.decide(features, explore=explore_gate)
        if decision.search:
            result = self.planner.search(observation, self.budget)
            action, transitions = int(result.action), int(result.model_transitions)
        else:
            p = np.asarray(probabilities, dtype=np.float64)
            action = int(self.rng.choice(p.size, p=p / p.sum())) if sample_action else int(np.argmax(p))
            result, transitions = None, 0
        return StepResult(action, decision.search, features, decision, transitions, perf_counter() - start, result)
