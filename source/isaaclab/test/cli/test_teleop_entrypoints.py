# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the unified teleoperation console entry point."""

import re
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest
import tomllib

import isaaclab.cli as cli

pytestmark = pytest.mark.unit

# Each teleop workflow and the repository-relative script it forwards to.
TELEOP_WORKFLOWS = {
    "run": ("scripts", "environments", "teleoperation", "teleop_se3_agent.py"),
    "record": ("scripts", "tools", "record_demos.py"),
    "replay": ("scripts", "tools", "replay_demos.py"),
}


@pytest.mark.parametrize(("command", "script_parts"), TELEOP_WORKFLOWS.items())
def test_teleop_workflow_help_exposes_task_preset_selectors(source_checkout_root: Path, command, script_parts):
    """Every teleop workflow accepts the task preset selectors documented for teleoperation."""
    script = source_checkout_root.joinpath(*script_parts)

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=source_checkout_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "preset selection:" in result.stdout
    assert "physics=NAME" in result.stdout
    assert "renderer=NAME" in result.stdout
    assert "presets=NAME[,NAME,...]" in result.stdout


@pytest.mark.parametrize(("command", "script_parts"), TELEOP_WORKFLOWS.items())
def test_teleop_dispatches_to_an_existing_script(source_checkout_root: Path, command, script_parts):
    """``isaaclab teleop`` forwards the remaining arguments to a script that exists."""
    args = [command, "--task", "IsaacContrib-PickPlace-Locomanipulation-G1-Abs", "--xr"]

    with (
        mock.patch.object(sys, "argv", ["isaaclab", "teleop", *args]),
        mock.patch("isaaclab.cli.run_python_command") as run_python,
    ):
        cli.cli()

    script = source_checkout_root.joinpath(*script_parts)
    run_python.assert_called_once_with(script, args[1:], check=True)
    assert script.is_file()


def _requirement_names(requirements: list[str]) -> set[str]:
    """Return the importable module names for a list of requirement strings."""
    names = set()
    for requirement in requirements:
        name = re.split(r"[\s<>=!~\[;@]", requirement, maxsplit=1)[0]
        if name.startswith("isaaclab"):
            names.add(name.replace("-", "_"))
    return names


@pytest.mark.parametrize(("command", "script_parts"), TELEOP_WORKFLOWS.items())
def test_teleop_workflow_isaaclab_imports_are_covered_by_the_extras(source_checkout_root: Path, command, script_parts):
    """Every ``isaaclab_*`` package a teleop script imports must ship with the extras.

    ``record_demos.py`` imports ``isaaclab_mimic`` at module level, so an environment built
    from ``--extra teleop`` alone used to die with ``ModuleNotFoundError`` only after Isaac Sim
    had finished booting. This catches that class of gap without needing an install.
    """
    with (source_checkout_root / "pyproject.toml").open("rb") as f:
        pyproject = tomllib.load(f)
    project = pyproject["project"]

    available = _requirement_names(project["dependencies"])
    available |= _requirement_names(project["optional-dependencies"]["teleop"])

    source = source_checkout_root.joinpath(*script_parts).read_text(encoding="utf-8")
    imported = set(re.findall(r"^\s*(?:import|from)\s+(isaaclab_\w+)", source, re.MULTILINE))

    missing = imported - available
    assert not missing, f"'isaaclab teleop {command}' imports {sorted(missing)}, absent from the teleop extra"


def test_teleop_rejects_unknown_workflow():
    """An unknown workflow name fails at the sub-parser instead of launching a script."""
    with (
        mock.patch.object(sys, "argv", ["isaaclab", "teleop", "bogus"]),
        mock.patch("isaaclab.cli.run_python_command") as run_python,
        pytest.raises(SystemExit),
    ):
        cli.cli()

    run_python.assert_not_called()


def test_teleop_environment_creation_failure_exits_nonzero_and_cleans_up(source_checkout_root: Path):
    """A failed Gym constructor must not look like a successful teleoperation session.

    Execute the script's actual run function in a subprocess, isolating its simulator-only
    imports so the failure/cleanup boundary can be tested without starting Kit.
    """
    script = source_checkout_root.joinpath(*TELEOP_WORKFLOWS["run"])
    code = """
import ast
import logging
import sys
from contextlib import ExitStack
from pathlib import Path
from types import ModuleType, SimpleNamespace

tree = ast.parse(Path(sys.argv[1]).read_text())
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_teleoperation")
annotations = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
module = ast.Module(body=[annotations, function], type_ignores=[])
ast.fix_missing_locations(module)

def close():
    print("CAMERA_FEED_CLOSED", flush=True)

def fail_make(*args, **kwargs):
    raise RuntimeError("gym constructor regression sentinel")

modules = {
    "isaaclab_physx.renderers.isaac_rtx_renderer_utils": {"apply_isaac_rtx_global_settings": None},
    "isaaclab_teleop": {"XrCameraFeedSession": SimpleNamespace(prepare=lambda *args, **kwargs:
                        SimpleNamespace(close=close))},
    "isaaclab.devices": {"Se3Keyboard": None, "Se3KeyboardCfg": None},
}
for name, attributes in modules.items():
    imported = ModuleType(name)
    vars(imported).update(attributes)
    sys.modules[name] = imported
namespace = {
    "args_cli": SimpleNamespace(teleop_device="keyboard", xr=False, disable_external_cameras=True, task="BrokenTask"),
    "gym": SimpleNamespace(make=fail_make),
    "logger": logging.getLogger("teleop-test"),
    "_rtx_rendering_requested": lambda args: False,
}
exec(compile(module, sys.argv[1], "exec"), namespace)
with ExitStack() as cleanup:
    namespace["run_teleoperation"](object(), False, cleanup)
"""
    result = subprocess.run([sys.executable, "-c", code, str(script)], capture_output=True, text=True, check=False)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "RuntimeError: gym constructor regression sentinel" in result.stderr
    assert "Traceback (most recent call last)" in result.stderr
    assert "CAMERA_FEED_CLOSED" in result.stdout
