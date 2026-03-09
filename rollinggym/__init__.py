# pylint: disable=C0103
"""
Registering the environment with Gymnasium.

you can use the following keywords here:
- reward_threshold
- nondeterministic
- max_episode_steps
- order_enforce     default: True
- kwargs            The default kwargs to pass to the environment class

"""
from __future__ import annotations

from gymnasium.envs.registration import register

register(
    id='rollinggym/FlatRolling-v0.0',
    entry_point='rollinggym.env:FlatRollingEnv',
)
