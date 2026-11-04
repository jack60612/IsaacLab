# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise Kit ray dispatch through real Newton substeps and graph replay.

The viewport adapter's gesture/keyboard subscriptions belong to Isaac Sim's
TestNewtonStagePicking suite. Only that UI dependency is replaced here; picking,
force bounds, solver execution, capture, and reset are real.
"""

import builtins
import sys
from types import ModuleType
from unittest.mock import Mock

import isaaclab_visualizers.kit.newton_picking as picking_module
import numpy as np
import pytest
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonBuilderCfg, NewtonCfg
from isaaclab_visualizers.kit.kit_visualizer_cfg import KitBackendCfg
from isaaclab_visualizers.kit.newton_picking import KitNewtonPicking

from isaaclab.sim import SimulationCfg, build_simulation_context

pytestmark = [pytest.mark.integration, pytest.mark.kitless]


@pytest.mark.parametrize(
    ("device", "capture", "substeps"),
    [("cpu", False, 1), ("cuda:0", True, 3)],
    ids=["uncaptured", "captured_substeps"],
)
def test_kit_rays_drive_newton_and_release_captured_forces(monkeypatch, device, capture, substeps):
    """Live rays move a body; hard reset, release and close neutralize replay."""
    interaction = Mock()
    monkeypatch.setattr(picking_module, "_load_viewport_interaction", lambda: Mock(return_value=interaction))
    sim_cfg = SimulationCfg(
        dt=0.005,
        device=device,
        gravity=(0.0, 0.0, 0.0),
        physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(), num_substeps=substeps, use_cuda_graph=capture),
    )
    cfg = KitBackendCfg(pick_max_acceleration=2.0)
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        builder = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=sim.cfg.physics))
        body = builder.add_body(mass=1.0)
        builder.add_shape_sphere(body, radius=0.1)
        sim.reset()
        # Installed as a pre-capture consumer, exactly as KitVisualizer does.
        picking = KitNewtonPicking(sim.backend_manager, cfg, 0.01, None)
        try:
            for reset in range(2):
                if reset:
                    old_picker = picking._picking
                    sim.reset()
                    picking.rebind()
                    assert not old_picker.is_picking()
                initial = sim.backend_manager.backend.state_0.body_q.numpy()[0, :3].copy()
                # Stage centimeters; deliberately non-unit direction.
                assert picking.handle_raycast((0.0, -100.0, 0.0), (0.0, 4.0, 0.0), True)
                assert picking._picking.is_picking()
                # Capture while picked but before moving the target.
                sim.step(render=False)
                assert bool(sim.backend_manager.backend.graphs) == capture
                assert picking.handle_raycast((100.0, -100.0, 0.0), (0.0, 4.0, 0.0), True)
                for _ in range(20):
                    before = sim.backend_manager.backend.state_0.body_qd.numpy()[0].copy()
                    sim.step(render=False)
                    after = sim.backend_manager.backend.state_0.body_qd.numpy()[0]
                    # COM linear velocity is the first three components in Newton.
                    acceleration = np.linalg.norm(after[:3] - before[:3]) / sim_cfg.dt
                    assert acceleration <= cfg.pick_max_acceleration * 9.81 + 1e-3
                position = sim.backend_manager.backend.state_0.body_q.numpy()[0, :3]
                assert position[0] > initial[0] + 0.01
                assert not picking.handle_raycast(None, None, False)
                linear_velocity = sim.backend_manager.backend.state_0.body_qd.numpy()[:, :3].copy()
                for _ in range(3):
                    sim.step(render=False)
                    np.testing.assert_array_equal(sim.backend_manager.backend.state_0.body_f.numpy(), 0.0)
                    np.testing.assert_allclose(
                        sim.backend_manager.backend.state_0.body_qd.numpy()[:, :3], linear_velocity, atol=1e-5
                    )
            # Closing during an active captured drag must be safe even after the
            # visualizer drops its reference: manager callback retains inputs.
            position_cm = sim.backend_manager.backend.state_0.body_q.numpy()[0, :3] * 100.0
            ray = (position_cm[0], position_cm[1] - 100.0, position_cm[2])
            picking.handle_raycast(ray, (0.0, 1.0, 0.0), True)
            assert picking._picking.is_picking()
            picking.handle_raycast((ray[0] + 100.0, ray[1], ray[2]), (0.0, 1.0, 0.0), True)
            picking.close()
            assert not picking.is_enabled()
            linear_velocity = sim.backend_manager.backend.state_0.body_qd.numpy()[:, :3].copy()
            for _ in range(3):
                sim.step(render=False)
                np.testing.assert_array_equal(sim.backend_manager.backend.state_0.body_f.numpy(), 0.0)
                np.testing.assert_allclose(
                    sim.backend_manager.backend.state_0.body_qd.numpy()[:, :3], linear_velocity, atol=1e-5
                )
            interaction.destroy.assert_called_once_with()
        finally:
            picking.close()


@pytest.mark.parametrize("adapter_present", [True, False], ids=["disabled_discoverable", "missing_fork_adapter"])
def test_adapter_loads_without_enabling_or_importing_stage_owner(monkeypatch, tmp_path, adapter_present):
    """Discovery alone suffices, and a stock extension gives an actionable error."""
    omni = ModuleType("omni")
    monkeypatch.setitem(sys.modules, "omni", omni)

    manager = Mock()
    manager.get_extensions.return_value = [{"id": "isaacsim.physics.newton-1.0.0", "enabled": False}]
    manager.get_extension_path.return_value = str(tmp_path)
    kit_module = ModuleType("omni.kit")
    app_module = ModuleType("omni.kit.app")
    app_module.get_app = lambda: Mock(get_extension_manager=lambda: manager)
    kit_module.app = app_module
    ext_module = ModuleType("omni.ext")
    ext_module.get_extension_name = lambda _id: "isaacsim.physics.newton"
    monkeypatch.setitem(sys.modules, "omni.kit", kit_module)
    monkeypatch.setitem(sys.modules, "omni.kit.app", app_module)
    monkeypatch.setitem(sys.modules, "omni.ext", ext_module)
    monkeypatch.setattr(omni, "kit", kit_module, raising=False)
    monkeypatch.setattr(omni, "ext", ext_module, raising=False)
    adapter = tmp_path / "isaacsim/physics/newton/impl/viewport_interaction.py"
    if adapter_present:
        adapter.parent.mkdir(parents=True)
        adapter.write_text("class NewtonViewportInteraction:\n    marker = 'standalone adapter'\n")
    original_import = builtins.__import__

    def forbid_stage_owner(name, *args, **kwargs):
        if name.startswith("isaacsim.physics.newton"):
            raise AssertionError("Stage-owner package must not be imported")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_stage_owner)
    if adapter_present:
        assert picking_module._load_viewport_interaction().marker == "standalone adapter"
    else:
        with pytest.raises(RuntimeError, match="pr/newton-viewport-drag"):
            picking_module._load_viewport_interaction()
    manager.set_extension_enabled.assert_not_called()
    manager.set_extension_enabled_immediate.assert_not_called()
