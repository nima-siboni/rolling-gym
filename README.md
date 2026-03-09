# Rolling Gym — Rolling Optimization Environment

A [Gymnasium](https://gymnasium.farama.org/) environment for simulating and optimizing the hot flat rolling process of S355 structural steel. Rolling Gym wraps the [PyRoll](https://pyroll-project.github.io/pyroll-docs/) physics engine to provide a step-by-step rolling simulation where an agent (human, heuristic, or learned) controls pass schedule parameters.

## Features

- **Gymnasium-compatible** — standard `reset()` / `step()` API, register as `rollinggym/FlatRolling-v0.0`
- **Physics-backed simulation** — each rolling pass is computed by the PyRoll engine (force, torque, temperature, microstructure)
- **Microstructure tracking** — models dynamic (DRX), metadynamic (MDRX), and static (SRX) recrystallization plus grain growth
- **Action masking** — invalid actions (e.g., reduction beyond physical limits) are masked out
- **Configurable** — Pydantic-based configuration for material, mill geometry, equipment limits, and process constraints

## Installation

```bash
# with uv (recommended)
uv sync --python 3.10

# or with pip
pip install -e .
```

For development dependencies (pytest, linting):

```bash
uv sync --python 3.10 --dev
```

## Quick Start

```python
import gymnasium as gym
import rollinggym  # registers the environment

from rollinggym.env.config_presets import create_s355_config

config = create_s355_config()
env = gym.make("rollinggym/FlatRolling-v0.0", env_config={"render_mode": None, "config": config})

obs, info = env.reset()

# Take a random valid action
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

## Environment Details

### Action Space

`MultiDiscrete([501, 121, 7])` with three dimensions:

| Dimension | Range | Physical meaning |
|-----------|-------|-----------------|
| Height reduction | 0 – 500 | 0.0 – 50.0 mm (0.1 mm resolution) |
| Interpass time | 1 – 120 | 1 – 120 seconds (0 is masked) |
| Rolling velocity | 1 – 6 | 5 – 30 m/min in 5 m/min steps (0 is masked) |

### Observation Space

A `Dict` with two keys:

- **`observations`**: z-score normalized state vector (10 continuous values)

| Index | Variable | Unit |
|-------|----------|------|
| 0 | Current thickness | mm |
| 1 | Step count | - |
| 2 | Height reduction limit | mm |
| 3 | Target thickness | mm |
| 4 | Rolling force | N |
| 5 | Rolling torque | Nm |
| 6 | Current temperature | K |
| 7 | Target temperature | K |
| 8 | Current grain size | um |
| 9 | Target grain size | um |

- **`action_mask`**: binary masks for each action dimension (1.0 = allowed, 0.0 = disallowed)

### Reward Structure

Each step incurs a fixed step penalty (-5.0) plus:

| Component | Description | Range |
|-----------|-------------|-------|
| Grain size progress | Rewards approaching target grain size, penalizes overshooting | asymmetric |
| HR efficiency | Rewards large reductions within equipment limits | [-5, 10] |
| Grain size completion | Bonus/penalty at episode end based on grain size accuracy | [-2, 25] |
| Temperature completion | Bonus/penalty at episode end based on final temperature | [-2, 25] |

Episodes terminate when the target thickness is reached, or truncate after 25 steps.

### Action Masking

Height reduction is constrained so that:
1. It does not exceed the per-pass HR limit
2. It does not exceed 70% of the current thickness
3. It does not reduce thickness below the target

Interpass time = 0 and rolling velocity = 0 are always masked.

## Configuration

Use factory functions to create validated configurations:

```python
from rollinggym.env.config_presets import create_s355_config

# Default S355 configuration
config = create_s355_config()

# Custom geometry
config = create_s355_config(
    initial_thickness_mm=100.0,
    target_thickness_mm=10.0,
)

# Override equipment limits
config = create_s355_config(
    force_limit_MN=4.0,
    torque_limit_kNm=130.0,
    power_limit_kW=400.0,
)
```

See `rollinggym/env/config_presets.py` for all available parameters.

## Examples

The `example_scripts/` directory contains runnable examples:

- `simulate_pass_schedule.py` — run a full pass schedule simulation
- `compare_interpass_scenarios.py` — compare different interpass time strategies
- `pyroll_example_pass_schedule.py` — direct PyRoll simulation example

## Running Tests

```bash
pytest
```

## Citation

If you use this software, please cite:

```bibtex
@software{rollinggym,
  author = {H. Siboni, Nima and Kiamousavi, Seyedreza},
  title = {Rolling Gym},
  url = {https://github.com/nima-siboni/rolling-gym},
}
```

## License

[Apache 2.0](LICENSE)
