# CS291K_Group2_Project

## Selective Search

A gate decides at each step whether to call the planner. It is trained to
balance task reward against search cost; PPO and the planner stay fixed.

- `selective_search.py`: `SelectiveSearchAgent` plus gates: `NeverGate` (plain PPO),
  `AlwaysGate` (Always-Search), `EntropyGate` (heuristic: search when PPO is unsure)
  and `LearnedGate` (small actor-critic MLP).
- `gate_training.py`: PPO training for `LearnedGate`, an episode runner for any
  Gymnasium-style env, and `evaluate` for comparing methods with the same metrics.

The gate's input is a summary of PPO's output: normalized entropy, top action
probability, top-2 margin, and critic value. `extra_features(obs)` can append more.
Its reward each step is `env_reward - search_cost * cost`, where cost is 1 per
planner call (`cost_unit="call"`) or the model transitions used
(`cost_unit="transition"`), and 0 when it skips search.

```python
from planning import MCTSPlanner, SearchBudget          # planning-search branch
from selective_search import LearnedGate, SelectiveSearchAgent
from gate_training import GateConfig, GateTrainer, evaluate, train_gate

budget = SearchBudget(simulations=32, max_depth=8)
planner = MCTSPlanner(dynamics, ppo_policy, action_count)
gate = LearnedGate()
agent = SelectiveSearchAgent(planner, budget, ppo_policy, gate)
trainer = GateTrainer(gate, GateConfig(search_cost=0.01))

train_gate(env, agent, trainer, iterations=200, episodes_per_iteration=16,
           log=lambda i, m: print(i, m))
print(evaluate(env, agent, trainer.config, episodes=100))
```

- The agent only needs `planner.search(obs, budget)` returning an object with
  `action` and `model_transitions`, and `ppo_policy(obs) -> (probabilities, value)`.
- `env` follows the Gymnasium API (`reset(seed=...)`, 5-tuple `step`). Success is
  counted as a positive reward on termination, which matches MiniGrid.
- Use the same `sample_action` setting for training and evaluation (default:
  PPO's most likely action).
- Sweep `search_cost` to trace the performance vs. compute trade-off.

Still to connect: the MiniGrid env, the PPO checkpoint/adapter and the learned
dynamics model. The tests use a toy corridor, a stub PPO policy and a stub planner.

```sh
pip install -r requirements.txt
python -m unittest discover -s tests -v
```
