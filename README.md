# CS291K_Group2_Project

## Planning & Search

`planning.py` implements PPO-guided Monte Carlo Tree Search for discrete actions.
The planner uses PUCT with PPO action probabilities as priors, the PPO critic at
nonterminal leaves, and discounted predicted rewards. It returns the most
visited root action. Ties use the PPO prior and then action order.

```python
from planning import AlwaysSearchAgent, MCTSPlanner, SearchBudget

budget = SearchBudget(simulations=32, max_depth=8)
planner = MCTSPlanner(dynamics, ppo_policy, action_count)
agent = AlwaysSearchAgent(planner, budget)
result = agent.act(observation)
```

- The PPO adapter implements `ppo_policy(observation) -> (probabilities, value)`.
  Probabilities must follow the environment's discrete action IDs, starting at 0.
  Use the same PPO checkpoint for the baseline and to construct the shared planner.
- The learned model implements `initial_state(observation) -> state` and
  `step(state, action) -> Prediction(state, observation, reward, done)`.
  The predicted observation must be usable by PPO. The model must preserve
  input states and support deterministic predictions. Include any time-limit
  information needed to predict episode boundaries in the model input/state.
- The selective gate can call `planner.search(observation, budget)` on gated
  decisions with the same budget as Always-Search. Both paths get `action`,
  `simulations`, `model_transitions`, `root_visits`, and `elapsed_seconds`.
  Search time includes model and policy inference within the call. The evaluator
  can count calls and sum transitions, recording zero on decisions that skip search.

The learned dynamics model and MiniGrid/PPO adapters still need to be connected.
The smoke tests use a toy dynamics stub and a fixed policy stub.

```sh
python -m unittest discover -s tests -v
```
