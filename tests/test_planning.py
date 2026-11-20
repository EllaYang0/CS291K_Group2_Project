"""Smoke tests with a toy dynamics stub and a fixed policy stub."""

import unittest

from planning import AlwaysSearchAgent, MCTSPlanner, Prediction, SearchBudget


class StubDynamics:
    """Action 1 advances twice to a goal. Action 0 ends without reward."""

    def __init__(self):
        self.calls = []

    def initial_state(self, observation):
        return observation

    def step(self, state, action):
        self.calls.append((state, action))
        if action == 0:
            return Prediction(state, state, 0.0, True)
        next_state = state + 1
        goal = next_state == 2
        return Prediction(next_state, next_state, float(goal), goal)


def stub_policy(observation):
    if observation == 2:
        raise AssertionError("terminal states must not use the critic")
    return (0.6, 0.4), 0.0


class PlanningSmokeTest(unittest.TestCase):
    def test_always_search_reaches_goal_against_policy_prior(self):
        model = StubDynamics()
        budget = SearchBudget(simulations=32, max_depth=2)
        planner = MCTSPlanner(model, stub_policy, action_count=2)
        agent = AlwaysSearchAgent(planner, budget)

        observation = 0
        search_calls = 0
        model_transitions = 0
        while observation < 2:
            calls_before = len(model.calls)
            result = agent.act(observation)
            search_calls += 1
            model_transitions += result.model_transitions
            self.assertEqual(result.action, 1)
            self.assertEqual(sum(result.root_visits), budget.simulations)
            self.assertEqual(result.model_transitions, len(model.calls) - calls_before)
            self.assertLessEqual(result.model_transitions, budget.simulations)
            observation += 1  # Apply action 1 to the toy environment.

        self.assertEqual(search_calls, 2)
        self.assertEqual(model_transitions, len(model.calls))

    def test_direct_search_honors_depth_limit(self):
        model = StubDynamics()
        planner = MCTSPlanner(model, stub_policy, action_count=2)
        result = planner.search(0, SearchBudget(simulations=16, max_depth=1))
        self.assertEqual(result.model_transitions, 2)
        self.assertTrue(all(state == 0 for state, _ in model.calls))


if __name__ == "__main__":
    unittest.main()
