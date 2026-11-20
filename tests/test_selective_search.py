"""Tests with a toy env, a stub PPO policy and a stub planner (no MiniGrid needed)."""

import os
import sys
import unittest
from dataclasses import dataclass

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gate_training import GateConfig, GateTrainer, evaluate, run_episode, train_gate  # noqa: E402
from selective_search import (  # noqa: E402
    AlwaysGate, EntropyGate, LearnedGate, NeverGate, SelectiveSearchAgent, policy_features,
)


class ForkCorridor:
    """Positions 0..length-1, goal at the end. Action 1 moves forward; action 0 ends the episode with 0 reward.

    Observation = position. The stub PPO is wrong (and unsure) only at the fork, position 0.
    """

    def __init__(self, length=4, time_limit=None):
        self.length, self.time_limit = length, time_limit

    def reset(self, seed=None):
        self.pos, self.t = 0, 0
        return self.pos, {}

    def step(self, action):
        self.t += 1
        if action == 0:
            return self.pos, 0.0, True, False, {}
        self.pos += 1
        goal = self.pos == self.length
        truncated = not goal and self.time_limit is not None and self.t >= self.time_limit
        return self.pos, float(goal), goal, truncated, {}


def fork_policy(obs):
    return ((0.6, 0.4), 0.5) if obs == 0 else ((0.05, 0.95), 0.9)


def confident_policy(obs):
    return (0.05, 0.95), 0.9


@dataclass(frozen=True)
class StubResult:
    action: int
    model_transitions: int


@dataclass(frozen=True)
class StubBudget:
    simulations: int = 8


class OraclePlanner:
    def __init__(self):
        self.calls = 0

    def search(self, observation, budget):
        self.calls += 1
        return StubResult(1, budget.simulations)


def make_agent(gate, policy=fork_policy, seed=0):
    return SelectiveSearchAgent(OraclePlanner(), StubBudget(), policy, gate, rng=np.random.default_rng(seed))


class FeatureTest(unittest.TestCase):
    def test_policy_features(self):
        uniform = policy_features((0.5, 0.5), 1.0)
        sure = policy_features((0.0, 1.0), 0.0)
        np.testing.assert_allclose(uniform, [1.0, 0.5, 0.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(sure, [0.0, 1.0, 1.0, 0.0], atol=1e-6)


class FixedGateTest(unittest.TestCase):
    def test_never_always_and_entropy_gates(self):
        env, config = ForkCorridor(), GateConfig()
        never = evaluate(env, make_agent(NeverGate()), config, episodes=3)
        always = evaluate(env, make_agent(AlwaysGate()), config, episodes=3)
        entropy = evaluate(env, make_agent(EntropyGate(0.5)), config, episodes=3)

        self.assertEqual(never["success_rate"], 0.0)  # greedy PPO takes the wrong turn at the fork
        self.assertEqual(always["success_rate"], 1.0)
        self.assertEqual(always["searches_per_episode"], 4)
        self.assertEqual(always["transitions_per_episode"], 32)
        self.assertEqual(entropy["success_rate"], 1.0)
        self.assertEqual(entropy["searches_per_episode"], 1)  # only the unsure fork

    def test_cost_accounting(self):
        agent = make_agent(AlwaysGate())
        calls = run_episode(ForkCorridor(), agent, GateConfig(search_cost=0.1, cost_unit="call"), train=False)
        trans = run_episode(ForkCorridor(), agent, GateConfig(search_cost=0.1, cost_unit="transition"), train=False)
        self.assertAlmostEqual(calls.gate_return, 1.0 - 0.1 * 4)
        self.assertAlmostEqual(trans.gate_return, 1.0 - 0.1 * 32)
        self.assertEqual(agent.planner.calls, 8)


class LearnedGateTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def search_prob(self, agent, obs):
        return agent.gate.decide(agent.features(obs)[0]).prob

    def test_learns_to_search_only_at_the_fork(self):
        gate = LearnedGate()
        agent = make_agent(gate)
        trainer = GateTrainer(gate, GateConfig(search_cost=0.05, lr=3e-3))
        train_gate(ForkCorridor(), agent, trainer, iterations=40, episodes_per_iteration=16, seed=0)

        self.assertGreater(self.search_prob(agent, 0), 0.8)
        for pos in (1, 2, 3):
            self.assertLess(self.search_prob(agent, pos), 0.2)
        result = evaluate(ForkCorridor(), agent, trainer.config, episodes=5)
        self.assertEqual(result["success_rate"], 1.0)
        self.assertEqual(result["searches_per_episode"], 1)

    def test_stops_searching_when_search_does_not_help(self):
        gate = LearnedGate()
        agent = make_agent(gate, policy=confident_policy)
        trainer = GateTrainer(gate, GateConfig(search_cost=0.2, lr=3e-3))
        train_gate(ForkCorridor(), agent, trainer, iterations=30, episodes_per_iteration=16, seed=0)
        self.assertLess(self.search_prob(agent, 0), 0.1)

    def test_time_limit_bootstraps_and_trains(self):
        gate = LearnedGate()
        agent = make_agent(gate)
        trainer = GateTrainer(gate, GateConfig())
        history = train_gate(ForkCorridor(time_limit=2), agent, trainer, iterations=2, episodes_per_iteration=4, seed=0)
        self.assertEqual(len(history), 2)
        self.assertTrue(all(np.isfinite(v) for m in history for v in m.values()))


if __name__ == "__main__":
    unittest.main()
