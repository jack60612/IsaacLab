# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for manager-based RL environments."""

from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from isaaclab.envs import ManagerBasedRLEnv

pytestmark = pytest.mark.unit


def _make_env_with_policy_obs_terms(
    num_envs: int,
    terms: list[tuple[str, tuple[int, ...], tuple[float, float] | None]],
) -> ManagerBasedRLEnv:
    """Build an uninitialized env whose observation manager stubs drive space setup.

    Args:
        num_envs: Number of vectorized environments.
        terms: Non-concatenated policy observation terms as ``(name, shape, clip)``
            where ``clip`` is ``(low, high)`` or ``None``.

    Returns:
        An uninitialized :class:`~isaaclab.envs.ManagerBasedRLEnv` ready for
        :meth:`~isaaclab.envs.ManagerBasedRLEnv._configure_gym_env_spaces`.
    """
    term_names = [name for name, _, _ in terms]
    term_dims = [shape for _, shape, _ in terms]
    term_cfgs = [SimpleNamespace(clip=clip) for _, _, clip in terms]

    env = object.__new__(ManagerBasedRLEnv)
    env._is_closed = True
    env.scene = SimpleNamespace(num_envs=num_envs)
    env.observation_manager = SimpleNamespace(
        active_terms={"policy": term_names},
        group_obs_concatenate={"policy": False},
        group_obs_dim={"policy": term_dims},
        _group_obs_term_cfgs={"policy": term_cfgs},
    )
    env.action_manager = SimpleNamespace(action_term_dim=[0])
    return env


def test_non_concatenated_obs_groups_contain_all_terms():
    """Non-concatenated observation groups expose every term in the Dict space (issue #3133).

    Before the fix, only the last term in each non-concatenated group would be present
    in the observation space Dict. This test ensures all terms are correctly included.
    """
    terms = [
        ("image", (4, 5, 3), (0.0, 1.0)),
        ("matrix", (2, 3), (-2.0, 2.0)),
        ("vector", (3,), None),
    ]
    env = _make_env_with_policy_obs_terms(num_envs=2, terms=terms)
    ManagerBasedRLEnv._configure_gym_env_spaces(env)

    assert isinstance(env.observation_space, gym.spaces.Dict)
    policy_space = env.observation_space.spaces["policy"]
    assert isinstance(policy_space, gym.spaces.Dict)

    expected_policy_terms = ["image", "matrix", "vector"]
    assert list(policy_space.spaces) == expected_policy_terms
    for term_name in expected_policy_terms:
        assert isinstance(policy_space.spaces[term_name], gym.spaces.Box)


def test_obs_space_follows_clip_constraint():
    """Observation space bounds reflect the clip constraint on each non-concatenated term."""
    terms = [
        ("vector", (3,), None),
        ("matrix", (2, 3), (-2.0, 2.0)),
        ("image", (4, 5, 3), (0.0, 1.0)),
    ]
    expected_shapes = {
        "vector": (2, 3),
        "matrix": (2, 2, 3),
        "image": (2, 4, 5, 3),
    }
    term_clips = {name: clip for name, _, clip in terms}
    env = _make_env_with_policy_obs_terms(num_envs=2, terms=terms)
    ManagerBasedRLEnv._configure_gym_env_spaces(env)

    for group_name, group_space in env.observation_space.spaces.items():
        assert isinstance(group_space, gym.spaces.Dict)
        for term_name, term_space in group_space.spaces.items():
            clip = term_clips[term_name]
            low = -np.inf if clip is None else clip[0]
            high = np.inf if clip is None else clip[1]
            assert isinstance(term_space, gym.spaces.Box), (
                f"Expected Box space for {term_name} in {group_name}, got {type(term_space)}"
            )
            assert term_space.shape == expected_shapes[term_name]
            assert np.all(term_space.low == low)
            assert np.all(term_space.high == high)


def _make_stepping_env(decimation: int, render_interval: int, physics_handles_decimation: bool):
    """Build an uninitialized env whose stubs record physics calls and render points."""
    env = object.__new__(ManagerBasedRLEnv)
    env._is_closed = True
    calls = SimpleNamespace(physics_steps=[], renders=[], backend_decimations=[])

    def sim_step(render):
        calls.physics_steps.append(env._step_decimation if physics_handles_decimation else 1)

    env.cfg = SimpleNamespace(
        decimation=decimation, sim=SimpleNamespace(dt=0.001, render_interval=render_interval), compute_final_obs=False
    )
    env.sim = SimpleNamespace(
        is_rendering=True,
        device="cpu",
        physics_manager=SimpleNamespace(set_decimation=calls.backend_decimations.append),
        step=sim_step,
        render=lambda skip_app_pumping: calls.renders.append(env._sim_step_counter),
        consume_reset_request=lambda: False,
    )
    env.scene = SimpleNamespace(num_envs=1, write_data_to_sim=lambda: None, update=lambda dt: None)
    env.action_manager = SimpleNamespace(process_action=lambda action: None, apply_action=lambda: None)
    env.recorder_manager = SimpleNamespace(
        active_terms=[], record_pre_step=lambda: None, record_post_physics_decimation_step=lambda: None
    )
    no_reset = torch.zeros(1, dtype=torch.bool)
    env.termination_manager = SimpleNamespace(compute=lambda: no_reset, terminated=no_reset, time_outs=no_reset)
    env.reward_manager = SimpleNamespace(compute=lambda dt: torch.zeros(1))
    env.command_manager = SimpleNamespace(compute=lambda dt: None)
    env.event_manager = SimpleNamespace(active_terms=[])
    env.observation_manager = SimpleNamespace(compute=lambda update_history=False: {})
    env.video_recorders = []
    env.episode_length_buf = torch.zeros(1, dtype=torch.long)
    env.common_step_counter = 0
    env.render_enabled = True
    env.extras = {}
    env._sim_step_counter = 3  # an earlier step left the counter off the render grid
    env._physics_handles_decimation = physics_handles_decimation
    env._step_decimation = decimation
    return env, calls


@pytest.mark.parametrize("physics_handles_decimation", [True, False])
def test_step_decimation_runs_each_step_to_its_own_length(physics_handles_decimation):
    """Alternating step lengths run exactly their physics steps and render on every crossed interval."""
    env, calls = _make_stepping_env(7, 20, physics_handles_decimation)
    lengths = [7, 7, 6] * 4
    for length in lengths:
        env.set_step_decimation(length)
        env.step(torch.zeros(1, 1))
    assert sum(calls.physics_steps) == sum(lengths) and env._sim_step_counter == 3 + sum(lengths)
    if physics_handles_decimation:
        assert calls.physics_steps == lengths
    # Only changes reach the backend, which keeps one captured loop per length.
    assert calls.backend_decimations == [6, 7, 6, 7, 6, 7, 6]
    # The counter visits 10, 17, 23, ...: every multiple of 20 crossed renders once, none twice.
    assert [count // 20 for count in calls.renders] == [1, 2, 3, 4]
    assert env.episode_length_buf.item() == env.common_step_counter == len(lengths)
    assert env.step_dt == env.cfg.sim.dt * env.cfg.decimation  # nominal step stays cfg.decimation


@pytest.mark.parametrize("decimation", [0, 8])
def test_step_decimation_rejects_lengths_outside_the_configured_bound(decimation):
    env, calls = _make_stepping_env(7, 20, True)
    with pytest.raises(ValueError, match="between 1 and 7"):
        env.set_step_decimation(decimation)
    assert env._step_decimation == 7 and not calls.backend_decimations
