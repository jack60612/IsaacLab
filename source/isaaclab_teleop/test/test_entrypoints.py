# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :mod:`isaaclab_teleop.entrypoints`."""

import contextlib
import importlib
import importlib.metadata
import sys
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("module_name", "cli_name"),
    [
        ("isaaclab_teleop.entrypoints.teleop_se3_agent", "run_teleop_se3_agent_cli"),
        ("isaaclab_teleop.entrypoints.record_demos", "run_record_demos_cli"),
        ("isaaclab_teleop.entrypoints.replay_demos", "run_replay_demos_cli"),
        ("isaaclab_teleop.entrypoints.teleop_replay_agent", "run_teleop_replay_agent_cli"),
    ],
)
def test_entrypoints_load_out_of_tree_tasks_before_parsing(monkeypatch, module_name, cli_name):
    """Tasks registered through ``isaaclab.tasks`` load before parsing; ``sys.argv`` and the exit code survive."""
    module = importlib.import_module(module_name)
    calls = []
    task_entry_point = mock.Mock()
    task_entry_point.load.side_effect = lambda: calls.append("load tasks")

    def parse_args(argv):
        calls.append("parse")
        sys.argv = [sys.argv[0], "hydra.override=1"]

    monkeypatch.setattr(sys, "argv", ["caller", "--caller-flag"])
    monkeypatch.setattr(module, "_parse_args", parse_args)
    monkeypatch.setattr(module, "main", lambda: calls.append("main") or 3)
    with mock.patch.object(importlib.metadata, "entry_points", return_value=[task_entry_point]) as entry_points:
        status = getattr(module, cli_name)(["--task", "Out-Of-Tree-Task"])

    entry_points.assert_called_once_with(group="isaaclab.tasks")
    assert calls == ["load tasks", "parse", "main"]
    assert sys.argv == ["caller", "--caller-flag"]
    # only the session replay agent reports an outcome; the other workflows always exit 0
    assert status == (3 if cli_name == "run_teleop_replay_agent_cli" else 0)


def _teleop_module(monkeypatch):
    """The teleop entry point with Kit-free camera-feed and RTX hooks for driving :func:`run_teleoperation`."""
    import isaaclab_teleop

    module = importlib.import_module("isaaclab_teleop.entrypoints.teleop_se3_agent")
    events = []
    feed = SimpleNamespace(
        requires_responsive_denoising=False,
        close=lambda: events.append("camera feed closed"),
        refresh=lambda: None,
        bind=lambda env: contextlib.nullcontext(),
    )
    monkeypatch.setattr(isaaclab_teleop, "XrCameraFeedSession", SimpleNamespace(prepare=lambda *a, **kw: feed))
    monkeypatch.setattr(module, "has_kit", lambda: False)
    monkeypatch.setattr(
        module,
        "args_cli",
        SimpleNamespace(
            task="Task",
            device="cpu",
            xr=False,
            disable_external_cameras=True,
            kit_visualizer=False,
            cloudxr_env=None,
            auto_launch_cloudxr=False,
            enable_debug_visualization=False,
        ),
        raising=False,
    )
    return module, events


def test_teleop_environment_creation_failure_propagates_and_cleans_up(monkeypatch):
    """A failed Gym constructor raises (non-zero exit) instead of ending like a successful session."""
    module, events = _teleop_module(monkeypatch)

    def fail_make(*args, **kwargs):
        raise RuntimeError("gym constructor regression sentinel")

    monkeypatch.setattr(module.gym, "make", fail_make)
    with pytest.raises(RuntimeError, match="gym constructor regression sentinel"):
        with ExitStack() as cleanup:
            module.run_teleoperation(SimpleNamespace(), object(), cleanup)
    assert events == ["camera feed closed"]


class _Flag:
    def __init__(self, value):
        self.value = value

    def any(self):
        return self.value


class _Action:
    def repeat(self, *args):
        return self


class _Env:
    num_envs = 1

    def __init__(self):
        self.loops, self.episode_steps, self.resets, self.log = 0, 0, 0, []
        self.scene = SimpleNamespace(env_prim_paths=["/World/envs/env_0"])
        self.sim = SimpleNamespace(is_running=self.is_running, render=lambda: None, visualizers=[])
        self.unwrapped = self

    def is_running(self):
        self.loops += 1
        return self.loops <= 8

    def step(self, actions):
        self.episode_steps += 1
        self.log.append(("step", self.resets))

    def reset(self):
        self.resets += 1
        self.episode_steps = 0
        self.log.append(("reset", self.resets))

    def close(self):
        pass


class _Device:
    def __init__(self, callbacks):
        self.callbacks, self.advances, self.resets = callbacks, 0, 0

    def advance(self):
        self.advances += 1
        if self.advances == 5:
            self.callbacks["R"]()  # the operator presses R after seeing success
        return _Action()

    def reset(self):
        self.resets += 1


def test_teleop_reports_success_and_waits_for_operator_reset(monkeypatch, capsys):
    """Success is reported once per episode and the episode continues until the operator resets.

    An automatic success reset inside ``env.step()`` hides a success state that occurs at a single
    step (e.g. an object released at the success instant). The operator's reset must reset both the
    environment and the teleop device, and re-arm the success report.
    """
    module, _ = _teleop_module(monkeypatch)
    env, devices = _Env(), []
    monkeypatch.setattr(module.gym, "make", lambda *args, **kwargs: env)

    def create_device(cfg, *, callbacks, **kwargs):
        devices.append(_Device(callbacks))
        return devices[-1]

    monkeypatch.setattr(module, "create_teleop_device", create_device)

    def picked(environment, threshold):
        # A latched success that becomes true on an episode's second step.
        return _Flag(environment.episode_steps >= threshold)

    success_term = SimpleNamespace(func=picked, params={"threshold": 2})
    with ExitStack() as cleanup:
        module.run_teleoperation(SimpleNamespace(), object(), cleanup, success_term)

    lines = capsys.readouterr().out.splitlines()
    successes = [index for index, line in enumerate(lines) if line.startswith("Success")]
    resets = [index for index, line in enumerate(lines) if line == "Environment reset complete"]
    # Steps 1-2 succeed at step 2; R is pressed during the fifth loop; steps 6-7 succeed again.
    assert len(successes) == 2 and len(resets) == 1
    assert successes[0] < resets[0] < successes[1]
    # Initial + operator reset, for both the environment and the device; none between success and R.
    assert env.resets == 2 and devices[0].resets == 2
    assert env.log[:7] == [("reset", 1), *[("step", 1)] * 5, ("reset", 2)]


def test_teleop_main_disables_automatic_success_reset_before_launch(monkeypatch):
    """``main`` prepares success like ``record_demos.py`` before the environment exists and hands it to the loop."""
    module = importlib.import_module("isaaclab_teleop.entrypoints.teleop_se3_agent")
    events = []
    cfg = module.ManagerBasedRLEnvCfg.__new__(module.ManagerBasedRLEnvCfg)
    cfg.sim = SimpleNamespace(device=None)
    cfg.scene = SimpleNamespace(num_envs=None)
    cfg.terminations = SimpleNamespace(time_out=object(), success="success-term")

    def prepare(env_cfg):
        events.append(("prepare", env_cfg.terminations.success))
        env_cfg.terminations.success = "inert"
        return "success-term"

    @contextlib.contextmanager
    def launch(env_cfg, args):
        events.append(("launch", env_cfg.terminations.success))
        yield

    monkeypatch.setattr(
        module,
        "args_cli",
        SimpleNamespace(task="Task", device="cpu", num_envs=1, teleop_device="keyboard", sensitivity=1.0, xr=False),
        raising=False,
    )
    monkeypatch.setattr(module, "resolve_task_config", lambda task, agent: (cfg, None))
    monkeypatch.setattr(module, "select_teleop_cfg", lambda *args: object())
    monkeypatch.setattr(module, "_prepare_success_term", prepare)
    monkeypatch.setattr(module, "launch_simulation", launch)
    monkeypatch.setattr(
        module,
        "run_teleoperation",
        lambda env_cfg, teleop_cfg, cleanup, success_term: events.append(("run", success_term)),
    )
    module.main()
    assert events == [("prepare", "success-term"), ("launch", "inert"), ("run", "success-term")]
