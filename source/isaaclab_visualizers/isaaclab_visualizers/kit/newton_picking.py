# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bind Kit's existing Shift-drag adapter to Newton's public bounded picker."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import warp as wp
from newton.viewer import Picking


def _load_viewport_interaction():
    """Load only the fork's standalone adapter, never its stage-owner package."""
    import omni.ext
    import omni.kit.app

    manager = omni.kit.app.get_app().get_extension_manager()
    for extension in manager.get_extensions():
        if omni.ext.get_extension_name(extension["id"]) != "isaacsim.physics.newton":
            continue
        root = Path(manager.get_extension_path(extension["id"]))
        path = root / "isaacsim/physics/newton/impl/viewport_interaction.py"
        if not path.is_file():
            raise RuntimeError(
                "Kit Newton picking requires the Isaac Sim pr/newton-viewport-drag extension override: "
                f"missing {path}. Keep the stage-owner extension disabled."
            )
        spec = importlib.util.spec_from_file_location("_isaaclab_newton_viewport_interaction", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.NewtonViewportInteraction
    raise RuntimeError(
        "Kit Newton picking requires a discoverable isaacsim.physics.newton fork extension override; "
        "supply its --ext-path without enabling the stage-owner extension."
    )


class KitNewtonPicking:
    """Own stable picking inputs across capture, reset, and viewport teardown.

    The manager retains ``apply_forces`` until its graph is destroyed. Closing
    the UI releases, but retains, the picker so a captured graph can safely replay
    neutral inputs. A hard reset rebinds before the manager captures a new graph.
    """

    def __init__(self, manager, cfg, meters_per_unit: float, visible_envs: list[int] | None):
        NewtonViewportInteraction = _load_viewport_interaction()

        self._manager = manager
        self._cfg = cfg
        self._meters_per_unit = meters_per_unit
        self._visible_envs = visible_envs
        self._backend = None
        self._picking = None
        self._retained_pickers = []
        self._closed = False
        self.rebind()
        self._interaction = NewtonViewportInteraction(self.is_enabled, self.handle_raycast)

    def is_enabled(self) -> bool:
        """Reject input after closing or replacing the native representation."""
        return not self._closed and self._backend is self._manager.backend

    def rebind(self) -> None:
        """Release old inputs and bind the current model before graph capture."""
        self.release()
        backend = self._manager.backend
        if backend is not self._backend:
            if self._picking is not None:
                # The previous graph may still exist until initialize_solver
                # invalidates it after this pre-capture hook returns.
                self._retained_pickers.append(self._picking)
            self._backend = backend
            self._picking = Picking(
                backend.model,
                pick_stiffness=self._cfg.pick_stiffness,
                pick_damping=self._cfg.pick_damping,
                pick_max_acceleration=self._cfg.pick_max_acceleration,
            )
            if self._visible_envs is not None:
                mask = [int(i in self._visible_envs) for i in range(backend.model.world_count)]
                self._picking.visible_worlds_mask = wp.array(mask, dtype=int, device=backend.model.device)
        self._manager.register_state_force_callback(self.apply_forces)

    def handle_raycast(self, origin, direction, has_input: bool) -> bool:
        """Forward USD world rays to the picker, preserving the displayed depth."""
        if not has_input or not self.is_enabled() or origin is None or direction is None:
            self.release()
            return False
        ray_origin = tuple(float(value) for value in origin)
        ray_direction = tuple(float(value) for value in direction)
        length = math.hypot(*ray_direction)
        if (
            not all(math.isfinite(value) for value in (*ray_origin, *ray_direction))
            or not math.isfinite(length)
            or length == 0.0
        ):
            self.release()
            return False
        ray_start = wp.vec3(*(value * self._meters_per_unit for value in ray_origin))
        ray_dir = wp.vec3(*(value / length for value in ray_direction))
        if self._picking.is_picking():
            self._picking.update(ray_start, ray_dir)
        else:
            self._picking.pick(self._backend.state_0, ray_start, ray_dir)
        return True

    def apply_forces(self, state) -> None:
        """Record the public picker's force kernel in every solver substep."""
        # Host callbacks run during capture, not replay. The old graph is gone
        # by the time this is called against the newly bound representation.
        self._retained_pickers.clear()
        if not self._closed:
            self._picking.apply_force(state)

    def release(self) -> None:
        """Neutralize device inputs, including graphs already captured."""
        if self._picking is not None:
            self._picking.release()

    def close(self) -> None:
        """Destroy gesture subscriptions without dangling captured pointers."""
        if self._closed:
            return
        self._closed = True
        self.release()
        self._interaction.destroy()
        self._interaction = None
