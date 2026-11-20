"""PPO-guided MCTS for deterministic, discrete-action dynamics models."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from math import isclose, isfinite, sqrt
from time import perf_counter
from typing import Generic, Protocol, TypeVar

State = TypeVar("State")
Observation = TypeVar("Observation")


@dataclass(frozen=True)
class Prediction(Generic[State, Observation]):
    state: State
    observation: Observation
    reward: float
    done: bool  # An episode boundary, including a modeled time limit.


class DynamicsModel(Protocol[State, Observation]):
    def initial_state(self, observation: Observation) -> State:
        """Encode the current observation as a model state."""
        ...

    def step(self, state: State, action: int) -> Prediction[State, Observation]:
        """Predict one transition without mutating the input state or live env."""
        ...


@dataclass(frozen=True)
class SearchBudget:
    simulations: int = 32
    max_depth: int = 8

    def __post_init__(self) -> None:
        if self.simulations < 1 or self.max_depth < 1:
            raise ValueError("simulations and max_depth must be positive")


@dataclass(frozen=True)
class SearchResult:
    action: int
    simulations: int
    model_transitions: int
    root_visits: tuple[int, ...]
    elapsed_seconds: float


@dataclass
class _Edge:
    prior: float
    visits: int = 0
    total_return: float = 0.0


@dataclass
class _Node(Generic[State]):
    state: State
    value: float
    edges: list[_Edge]
    children: dict[int, tuple[float, "_Node[State]"]] = field(default_factory=dict)

    def select(self, exploration: float) -> int:
        scale = exploration * sqrt(1 + sum(edge.visits for edge in self.edges))

        def score(action: int) -> float:
            edge = self.edges[action]
            mean = edge.total_return / edge.visits if edge.visits else 0.0
            return mean + scale * edge.prior / (1 + edge.visits)

        return max(range(len(self.edges)), key=score)


class MCTSPlanner(Generic[State, Observation]):
    """Use PPO action probabilities as PUCT priors and its critic at leaves.

    policy(observation) returns (probabilities in action-ID order, scalar value).
    Each simulation expands at most one model transition. Predicted transitions
    are cached within a call, so this planner assumes deterministic predictions.
    """

    def __init__(
        self,
        dynamics: DynamicsModel[State, Observation],
        policy: Callable[[Observation], tuple[Sequence[float], float]],
        action_count: int,
        *,
        discount: float = 0.99,
        exploration: float = 1.0,
    ) -> None:
        if action_count < 1 or not 0 <= discount <= 1:
            raise ValueError("expected a positive action count and discount in [0, 1]")
        if not isfinite(exploration) or exploration <= 0:
            raise ValueError("exploration must be finite and positive")
        self.dynamics = dynamics
        self.policy = policy
        self.action_count = action_count
        self.discount = discount
        self.exploration = exploration

    def _node(self, state: State, observation: Observation) -> _Node[State]:
        probabilities, value = self.policy(observation)
        if (
            len(probabilities) != self.action_count
            or any(not isfinite(p) or p < 0 for p in probabilities)
            or not isclose(sum(probabilities), 1.0, abs_tol=1e-6)
            or not isfinite(value)
        ):
            raise ValueError("policy must return action probabilities and a finite value")
        return _Node(state, float(value), [_Edge(float(p)) for p in probabilities])

    def search(self, observation: Observation, budget: SearchBudget) -> SearchResult:
        start = perf_counter()
        root = self._node(self.dynamics.initial_state(observation), observation)
        transitions = 0

        for _ in range(budget.simulations):
            node = root
            path: list[tuple[_Edge, float]] = []
            for _ in range(budget.max_depth):
                if not node.edges:
                    break
                action = node.select(self.exploration)
                edge = node.edges[action]
                child = node.children.get(action)
                if child is not None:
                    reward, node = child
                    path.append((edge, reward))
                    continue

                prediction = self.dynamics.step(node.state, action)
                transitions += 1
                child_node = (
                    _Node(prediction.state, 0.0, [])
                    if prediction.done
                    else self._node(prediction.state, prediction.observation)
                )
                node.children[action] = (prediction.reward, child_node)
                path.append((edge, prediction.reward))
                node = child_node
                break

            value = node.value
            for edge, reward in reversed(path):
                value = reward + self.discount * value
                edge.visits += 1
                edge.total_return += value

        visits = tuple(edge.visits for edge in root.edges)
        action = max(
            range(self.action_count),
            key=lambda i: (visits[i], root.edges[i].prior),
        )
        return SearchResult(action, budget.simulations, transitions, visits, perf_counter() - start)


class AlwaysSearchAgent(Generic[State, Observation]):
    """Invoke the shared planner at every environment decision."""

    def __init__(self, planner: MCTSPlanner[State, Observation], budget: SearchBudget):
        self.planner = planner
        self.budget = budget

    def act(self, observation: Observation) -> SearchResult:
        return self.planner.search(observation, self.budget)
