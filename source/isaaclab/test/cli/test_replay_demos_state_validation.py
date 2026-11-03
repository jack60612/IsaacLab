# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check replay validation against the public EpisodeData sample shape."""

import ast
from pathlib import Path

import pytest
import torch

from isaaclab.utils.datasets import EpisodeData

pytestmark = pytest.mark.integration

_REPLAY_DEMOS_PATH = Path(__file__).resolve().parents[4] / "scripts" / "tools" / "replay_demos.py"


def _load_compare_states():
    # Follow the existing replay-loop test without launching Kit at script import.
    tree = ast.parse(_REPLAY_DEMOS_PATH.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "compare_states")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(_REPLAY_DEMOS_PATH), "exec"), namespace)
    return namespace["compare_states"]


def _recorded_and_runtime_states(runtime_env_index):
    recorded = {
        "articulation": {
            "robot": {
                "root_pose": torch.tensor([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]),
                "root_velocity": torch.arange(6, dtype=torch.float32),
                "joint_position": torch.arange(9, dtype=torch.float32),
            }
        },
        "rigid_object": {"cube": {"root_pose": torch.tensor([0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0])}},
    }
    episode = EpisodeData()
    episode.add("states", recorded)
    episode.pre_export()
    dataset_state = episode.get_state(0)
    assert dataset_state["articulation"]["robot"]["root_pose"].shape == (1, 7)
    runtime = {}
    for asset_type, assets in recorded.items():
        runtime[asset_type] = {}
        for asset_name, fields in assets.items():
            runtime[asset_type][asset_name] = {}
            for field_name, value in fields.items():
                batch = value.repeat(runtime_env_index + 1, 1) + 1.0
                batch[runtime_env_index] = value
                runtime[asset_type][asset_name][field_name] = batch
    return dataset_state, runtime


@pytest.mark.parametrize("runtime_env_index", [0, 1])
@pytest.mark.parametrize("different_value", [False, True])
def test_replay_compares_single_sample_to_selected_runtime_env(runtime_env_index, different_value):
    dataset_state, runtime = _recorded_and_runtime_states(runtime_env_index)
    if different_value:
        runtime["rigid_object"]["cube"]["root_pose"][runtime_env_index, 0] += 0.02
    matched, log = _load_compare_states()(dataset_state, runtime, runtime_env_index)
    assert matched is not different_value
    assert bool(log) is different_value


def test_replay_still_rejects_different_feature_counts():
    dataset_state, runtime = _recorded_and_runtime_states(0)
    runtime["articulation"]["robot"]["joint_position"] = torch.zeros(1, 8)
    with pytest.raises(ValueError, match="State shape of joint_position for asset robot"):
        _load_compare_states()(dataset_state, runtime, 0)
