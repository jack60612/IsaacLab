# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional, lockstep ROS 2 joint-position bridge for single manager-based environments.

ROS packages are imported only when constructing a bridge. Source the external ROS 2
Jazzy installation in a Python 3.12 environment before running the zero agent with
``--ros2``. No ROS timer or background executor advances the simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class Ros2BridgeCfg:
    """Task-owned contract exposed as ``env.cfg.ros2_bridge``.

    Joint names are exact articulation names in the desired publication order.
    Action terms must collectively control exactly those joints with absolute,
    unit-scale, zero-offset joint-position actions. Unselected terms must have
    zero width. Body names select world-frame transforms published on ``/tf``.
    """

    articulation_name: str
    joint_names: tuple[str, ...]
    action_terms: tuple[str, ...]
    joint_state_topic: str = "/joint_states"
    joint_command_topic: str = "/joint_commands"
    body_names: tuple[str, ...] = ()
    world_frame: str = "world"
    frame_prefix: str = ""
    """Prefix for published TF child frames, e.g. ``sim/`` to isolate measured frames."""
    exchange_period_s: float = 0.0
    """Integrated-time interval between exchanges; zero exchanges every environment step."""


def _tensor(value: Any) -> torch.Tensor:
    """Access either an Isaac Lab array's Torch view or an ordinary tensor."""
    return value if isinstance(value, torch.Tensor) else value.torch


class Ros2Bridge:
    """Own a ROS node, persistent position targets and reusable state messages.

    Call :meth:`advance` immediately before each environment step and
    :meth:`publish` immediately afterward. Commands are named, partial absolute
    positions; unspecified joints hold their last target (initially reset state).
    Invalid commands are rejected atomically with a warning, not an exception in
    the subscription callback. Optional velocity and effort arrays are validated
    for JointState compatibility but ignored by this position-only action bridge.
    """

    def __init__(self, env: Any):
        from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction

        self._env = env.unwrapped
        self.cfg = self._env.cfg.ros2_bridge
        if not isinstance(self.cfg, Ros2BridgeCfg):
            raise ValueError("env.cfg.ros2_bridge must be a Ros2BridgeCfg.")
        period = self.cfg.exchange_period_s
        if not math.isfinite(period) or period < 0.0:
            raise ValueError("ros2_bridge.exchange_period_s must be finite and nonnegative.")
        self._exchange_period_s = period
        self._exchange_origin: float | None = None
        self._next_exchange = 1
        self._exchange_pending = True
        if self._env.num_envs != 1:
            raise ValueError("The ROS 2 bridge requires exactly one environment.")
        for field in ("joint_names", "action_terms", "body_names"):
            names = getattr(self.cfg, field)
            if len(set(names)) != len(names) or any(not isinstance(name, str) or not name for name in names):
                raise ValueError(f"ros2_bridge.{field} must contain unique, nonempty names.")
        if not self.cfg.joint_names or not self.cfg.action_terms:
            raise ValueError("The ROS 2 bridge requires joint_names and action_terms.")
        if not self.cfg.world_frame:
            raise ValueError("ros2_bridge.world_frame must not be empty.")
        self._asset = self._env.scene[self.cfg.articulation_name]
        joint_lookup = {name: index for index, name in enumerate(self._asset.joint_names)}
        body_lookup = {name: index for index, name in enumerate(self._asset.body_names)}
        try:
            joint_ids = [joint_lookup[name] for name in self.cfg.joint_names]
            self._body_ids = [body_lookup[name] for name in self.cfg.body_names]
        except KeyError as exc:
            raise ValueError(f"Unknown bridge joint or body: {exc.args[0]}") from None
        manager = self._env.action_manager
        if set(self.cfg.action_terms) - set(manager.active_terms):
            raise ValueError("ros2_bridge.action_terms contains inactive action terms.")
        self._actions = torch.zeros_like(manager.action)
        self._command_indices: dict[str, int] = {}
        action_index = 0
        for name in manager.active_terms:
            term = manager.get_term(name)
            if name not in self.cfg.action_terms:
                if term.action_dim:
                    raise ValueError(f"Unselected action term {name!r} must have zero width.")
            else:
                if not isinstance(term, JointPositionAction) or term._asset is not self._asset:
                    raise ValueError(
                        f"Action term {name!r} must be an absolute JointPositionAction on the bridge asset."
                    )
                if (
                    term.cfg.use_default_offset
                    or not bool(torch.all(torch.as_tensor(term._scale) == 1))
                    or not bool(torch.all(torch.as_tensor(term._offset) == 0))
                    or term.cfg.clip is not None
                ):
                    raise ValueError(f"Action term {name!r} requires scale=1, offset=0, no default offset or clipping.")
                term_ids = term._joint_ids
                if isinstance(term_ids, slice):
                    term_ids = list(range(self._asset.num_joints))[term_ids]
                else:
                    term_ids = _tensor(term_ids).tolist() if not isinstance(term_ids, (list, tuple)) else term_ids
                if len(term_ids) != term.action_dim:
                    raise ValueError(f"Action term {name!r} has an inconsistent joint mapping.")
                for offset, joint_id in enumerate(term_ids):
                    joint_name = self._asset.joint_names[joint_id]
                    if joint_name not in self.cfg.joint_names or joint_name in self._command_indices:
                        raise ValueError(f"Action joint {joint_name!r} is unconfigured or controlled more than once.")
                    self._command_indices[joint_name] = action_index + offset
            action_index += term.action_dim
        if set(self._command_indices) != set(self.cfg.joint_names):
            raise ValueError("Action terms must control every configured bridge joint exactly once.")
        positions = _tensor(self._asset.data.joint_pos)
        # Callbacks mutate CPU targets only. A blocking copy after the ready drain
        # keeps this reusable staging memory safe to change at the next exchange.
        self._command_buffer = torch.empty_like(self._actions, device="cpu")
        initial_positions = positions[0].cpu()
        for name, index in self._command_indices.items():
            self._command_buffer[0, index] = initial_positions[joint_lookup[name]]
        if not bool(torch.isfinite(self._command_buffer).all()):
            raise ValueError("Initial bridge joint positions must be finite.")
        self._actions.copy_(self._command_buffer)
        self._commands_dirty = False
        self._joint_ids = torch.tensor(joint_ids, device=positions.device, dtype=torch.long)
        self._body_indices = torch.tensor(self._body_ids, device=positions.device, dtype=torch.long)
        joint_size = 3 * len(joint_ids)
        snapshot_size = joint_size + 7 * len(self._body_ids)
        self._gather_buffer = torch.empty(snapshot_size, dtype=positions.dtype, device=positions.device)
        self._snapshot_buffer = torch.empty(snapshot_size, dtype=positions.dtype, device="cpu")
        self._joint_gather = self._gather_buffer[:joint_size].view(3, len(joint_ids))
        self._pose_gather = self._gather_buffer[joint_size:].view(len(self._body_ids), 7)
        self._state_buffer = self._snapshot_buffer[:joint_size].view(3, len(joint_ids))
        self._pose_buffer = self._snapshot_buffer[joint_size:].view(len(self._body_ids), 7)
        self._closed = False

        # Keep the optional dependency entirely outside ordinary zero-agent use.
        import rclpy
        from geometry_msgs.msg import TransformStamped
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor, TimeoutException
        from rclpy.parameter import Parameter
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import JointState
        from tf2_msgs.msg import TFMessage

        self._context = Context()
        self._timeout_exception = TimeoutException
        self._node = None
        self._executor = None
        try:
            rclpy.init(context=self._context)
            self._executor = SingleThreadedExecutor(context=self._context)
            self._node = rclpy.create_node(
                "isaaclab_joint_bridge",
                context=self._context,
                parameter_overrides=[Parameter("use_sim_time", value=True)],
            )
            self._executor.add_node(self._node)
            self._state_pub = self._node.create_publisher(JointState, self.cfg.joint_state_topic, 10)
            self._clock_pub = self._node.create_publisher(Clock, "/clock", 10)
            self._tf_pub = self._node.create_publisher(TFMessage, "/tf", 10)
            self._subscription = self._node.create_subscription(
                JointState, self.cfg.joint_command_topic, self._on_command, 10
            )
            self._state_msg = JointState()
            self._state_msg.name = list(self.cfg.joint_names)
            self._state_msg.position = [0.0] * len(joint_ids)
            self._state_msg.velocity = [0.0] * len(joint_ids)
            self._state_msg.effort = [0.0] * len(joint_ids)
            self._clock_msg = Clock()
            self._tf_msg = TFMessage()
            for name in self.cfg.body_names:
                transform = TransformStamped()
                transform.header.frame_id = self.cfg.world_frame
                transform.child_frame_id = self.cfg.frame_prefix + name
                self._tf_msg.transforms.append(transform)
        except BaseException:
            self.close()
            raise

    def _on_command(self, message: Any) -> None:
        try:
            names = message.name
            values = message.position
            valid = (
                len(names) > 0
                and len(names) == len(values)
                and len(set(names)) == len(names)
                and all(name in self._command_indices for name in names)
                and all(
                    len(field) in (0, len(names)) and all(math.isfinite(value) for value in field)
                    for field in (message.velocity, message.effort)
                )
                and all(math.isfinite(value) for value in values)
            )
            # Validate representability in the action dtype before changing any target.
            limit = torch.finfo(self._actions.dtype).max
            valid = valid and all(abs(value) <= limit for value in values)
        except (AttributeError, TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            self._node.get_logger().warning("Rejected malformed ROS joint-position command.")
            return
        for name, value in zip(names, values):
            self._command_buffer[0, self._command_indices[name]] = value
        self._commands_dirty = True

    def advance(self) -> torch.Tensor:
        """Process commands only when an exchange is due; otherwise return held targets."""
        if self._exchange_period_s:
            elapsed = self._env.sim.backend_manager.get_simulation_time()
            if self._exchange_origin is None:
                self._exchange_origin = elapsed
            elif elapsed < self._exchange_origin + self._next_exchange * self._exchange_period_s:
                return self._actions
            else:
                # Anchor to the initial publication, not the quantized physics grid.
                # Skip missed intervals after a time jump instead of replaying a burst.
                self._next_exchange = max(
                    self._next_exchange + 1,
                    math.floor((elapsed - self._exchange_origin) / self._exchange_period_s) + 1,
                )
            self._exchange_pending = True
        # Each hardware system can publish its own partial command. Drain all ready
        # callbacks so arm and gripper targets reach the same physics step, rather
        # than letting one spin_once per exchange accumulate a stale command queue.
        dispatched = False
        while True:
            try:
                handler, _, _ = self._executor.wait_for_ready_callbacks(timeout_sec=0.0)
            except self._timeout_exception:
                # Jazzy's cached readiness generator can time out after a pass
                # that yielded callbacks, even with more messages on that entity.
                if not dispatched:
                    break
                dispatched = False
                continue
            handler()
            handler.result()  # Propagate callback failures, as SingleThreadedExecutor does.
            dispatched = True
        if self._commands_dirty:
            self._actions.copy_(self._command_buffer)
            self._commands_dirty = False
        return self._actions

    def publish(self) -> None:
        """Publish measured joints, world transforms and the actual simulation clock."""
        if self._exchange_period_s and not self._exchange_pending:
            return
        # Backend time advances only after integration and already includes decimation.
        # SimulationContext's step counter also counts paused/stopped attempts.
        elapsed = self._env.sim.backend_manager.get_simulation_time()
        if self._exchange_origin is None:
            self._exchange_origin = elapsed
        self._exchange_pending = False
        nanoseconds = round(elapsed * 1_000_000_000)
        sec, nanosec = divmod(nanoseconds, 1_000_000_000)
        for stamp in (self._clock_msg.clock, self._state_msg.header.stamp):
            stamp.sec = sec
            stamp.nanosec = nanosec
        self._state_msg.header.frame_id = self.cfg.world_frame
        data = self._asset.data
        for index, field in enumerate(("joint_pos", "joint_vel", "applied_torque")):
            torch.index_select(_tensor(getattr(data, field))[0], 0, self._joint_ids, out=self._joint_gather[index])
        if self._body_ids:
            torch.index_select(_tensor(data.body_link_pose_w)[0], 0, self._body_indices, out=self._pose_gather)
        # One contiguous device-to-host snapshot contains every published value.
        self._snapshot_buffer.copy_(self._gather_buffer)
        for field, row in zip(("position", "velocity", "effort"), self._state_buffer):
            values = getattr(self._state_msg, field)
            for index in range(len(values)):
                values[index] = float(row[index])
        if self._body_ids:
            for transform, pose in zip(self._tf_msg.transforms, self._pose_buffer):
                transform.header.stamp.sec = sec
                transform.header.stamp.nanosec = nanosec
                translation = transform.transform.translation
                rotation = transform.transform.rotation
                translation.x, translation.y, translation.z = (float(value) for value in pose[:3])
                # ProxyArray's wp.transformf Torch view stores quaternions in xyzw order.
                rotation.x, rotation.y, rotation.z, rotation.w = (float(value) for value in pose[3:])
        self._clock_pub.publish(self._clock_msg)
        self._state_pub.publish(self._state_msg)
        if self._body_ids:
            self._tf_pub.publish(self._tf_msg)

    def close(self) -> None:
        """Release this bridge's node and private ROS context; safe to call repeatedly."""
        if self._closed:
            return
        self._closed = True
        try:
            try:
                if self._executor is not None:
                    self._executor.shutdown()
            finally:
                if self._node is not None:
                    self._node.destroy_node()
        finally:
            self._context.try_shutdown()
