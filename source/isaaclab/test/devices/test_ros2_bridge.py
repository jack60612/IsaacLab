# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU contracts at the bridge boundary; no simulator or ROS installation required."""

import copy
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from isaaclab.devices.ros2 import Ros2Bridge, Ros2BridgeCfg

pytestmark = pytest.mark.kitless


def _header():
    return SimpleNamespace(stamp=SimpleNamespace(sec=0, nanosec=0), frame_id="")


class JointState:
    def __init__(self):
        self.header = _header()
        self.name = []
        self.position = []
        self.velocity = []
        self.effort = []


class TransformStamped:
    def __init__(self):
        self.header = _header()
        self.child_frame_id = ""
        self.transform = SimpleNamespace(
            translation=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0),
        )


class JointPositionAction:
    def __init__(self, asset, ids):
        self._asset = asset
        self._joint_ids = torch.tensor(ids)
        self.action_dim = len(ids)
        self._scale = 1.0
        self._offset = 0.0
        self.cfg = SimpleNamespace(use_default_offset=False, clip=None)


@pytest.fixture
def bridge_env(monkeypatch, bridge_model):
    """Provide transport doubles only; production code owns mapping and state conversion."""
    node = SimpleNamespace(published={}, pending=[], warnings=[], destroyed=False, spins=0)

    def publisher(_kind, topic, _depth):
        node.published[topic] = []
        return SimpleNamespace(publish=lambda msg: node.published[topic].append(copy.deepcopy(msg)))

    def subscription(_kind, _topic, callback, _depth):
        node.callback = callback
        return object()

    node.create_publisher = publisher
    node.create_subscription = subscription
    node.get_logger = lambda: SimpleNamespace(warning=node.warnings.append)
    node.destroy_node = lambda: setattr(node, "destroyed", True)
    context = SimpleNamespace(shutdown=False)
    context.try_shutdown = lambda: setattr(context, "shutdown", True)

    class TimeoutException(Exception):
        pass

    class ReadyCallback:
        def __call__(self):
            message = node.pending.pop(0)
            if callable(message):
                message()
            else:
                node.callback(message)

        def result(self):
            return None

    class Executor:
        def __init__(self, *, context):
            assert context is bridge_context
            self.node = None
            self.stopped = False
            self.yielded = False

        def add_node(self, value):
            self.node = value

        def wait_for_ready_callbacks(self, *, timeout_sec):
            assert self.node is node and timeout_sec == 0.0
            node.spins += 1
            # Match Jazzy's cached generator: timeout after a yielded readiness
            # pass does not imply the subscription's message queue is empty.
            if self.yielded:
                self.yielded = False
                raise TimeoutException
            if not node.pending:
                raise TimeoutException
            self.yielded = True
            return ReadyCallback(), None, node

        def shutdown(self):
            self.stopped = True

    bridge_context = context

    modules = {
        "rclpy": dict(init=lambda **kwargs: None, create_node=lambda *args, **kwargs: node),
        "rclpy.context": dict(Context=lambda: context),
        "rclpy.executors": dict(SingleThreadedExecutor=Executor, TimeoutException=TimeoutException),
        "rclpy.parameter": dict(Parameter=lambda name, *, value: SimpleNamespace(name=name, value=value)),
        "sensor_msgs.msg": dict(JointState=JointState),
        "geometry_msgs.msg": dict(TransformStamped=TransformStamped),
        "rosgraph_msgs.msg": dict(Clock=lambda: SimpleNamespace(clock=SimpleNamespace(sec=0, nanosec=0))),
        "tf2_msgs.msg": dict(TFMessage=lambda: SimpleNamespace(transforms=[])),
        "isaaclab.envs.mdp.actions.joint_actions": dict(JointPositionAction=JointPositionAction),
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    bridge_model.node = node
    bridge_model.context = context
    yield bridge_model


@pytest.fixture
def bridge_model(monkeypatch):
    module = ModuleType("isaaclab.envs.mdp.actions.joint_actions")
    module.JointPositionAction = JointPositionAction
    monkeypatch.setitem(sys.modules, module.__name__, module)
    data = SimpleNamespace(
        joint_pos=SimpleNamespace(torch=torch.tensor([[1.0, 2.0, 3.0]])),
        joint_vel=torch.tensor([[0.1, 0.2, 0.3]]),
        applied_torque=SimpleNamespace(torch=torch.tensor([[10.0, 20.0, 30.0]])),
        body_link_pose_w=SimpleNamespace(
            torch=torch.tensor(
                [
                    [
                        [100.0, 101.0, 102.0, 0.0, 0.0, 0.0, 1.0],
                        [4.0, 5.0, 6.0, 0.5, 0.1, 0.2, 0.3],
                        [200.0, 201.0, 202.0, 0.0, 0.0, 0.0, 1.0],
                    ]
                ]
            )
        ),
    )
    asset = SimpleNamespace(joint_names=["a", "b", "c"], body_names=["plant", "tool", "apple"], num_joints=3, data=data)
    terms = {
        "mechanics": SimpleNamespace(action_dim=0),
        "arm": JointPositionAction(asset, [2, 0]),
        "gripper": JointPositionAction(asset, [1]),
    }
    cfg = Ros2BridgeCfg("robot", ("b", "c", "a"), ("gripper", "arm"), body_names=("tool",))
    env = SimpleNamespace(
        cfg=SimpleNamespace(ros2_bridge=cfg, decimation=4),
        num_envs=1,
        scene={"robot": asset},
        action_manager=SimpleNamespace(action=torch.zeros(1, 3), active_terms=list(terms), get_term=terms.__getitem__),
        sim=SimpleNamespace(steps=0, time=0.0),
    )
    env.unwrapped = env
    env.sim.get_physics_step_count = lambda: env.sim.steps
    env.sim.get_physics_dt = lambda: 0.0005
    env.sim.backend_manager = SimpleNamespace(
        handles_decimation=lambda: False, get_simulation_time=lambda: env.sim.time
    )
    return SimpleNamespace(env=env, asset=asset, terms=terms)


def _command(names, values):
    message = JointState()
    message.name = names
    message.position = values
    return message


def test_named_partial_commands_hold_and_follow_resolved_action_order(bridge_env):
    bridge = Ros2Bridge(bridge_env.env)
    try:
        actions = bridge.advance()
        torch.testing.assert_close(actions, torch.tensor([[3.0, 1.0, 2.0]]))
        bridge_env.node.pending.append(_command(["b", "c"], [8.0, 9.0]))
        assert bridge.advance() is actions
        torch.testing.assert_close(actions, torch.tensor([[9.0, 1.0, 8.0]]))
        bridge_env.node.pending.append(_command(["a"], [7.0]))
        bridge.advance()
        bridge_env.asset.data.joint_pos.torch.zero_()  # episode reset must not erase held targets
        torch.testing.assert_close(bridge.advance(), torch.tensor([[9.0, 7.0, 8.0]]))
    finally:
        bridge.close()
        bridge.close()
    assert bridge_env.node.destroyed and bridge_env.context.shutdown


@pytest.mark.parametrize(
    "names,values,extra",
    [
        (["a", "unknown"], [4.0, 5.0], {}),
        (["a", "a"], [4.0, 5.0], {}),
        (["a", "b"], [4.0], {}),
        ([], [], {}),
        (["a", "b"], [4.0, float("nan")], {}),
        (["a"], [float("inf")], {}),
        (["a"], [1e100], {}),
        (["a"], ["invalid"], {}),
        (["a"], [4.0], {"velocity": [1.0, 2.0]}),
        (["a"], [4.0], {"effort": [float("nan")]}),
        (["a"], [4.0], {"velocity": [float("inf")]}),
        (["a"], [4.0], {"effort": ["invalid"]}),
    ],
)
def test_invalid_commands_are_rejected_atomically(bridge_env, names, values, extra):
    bridge = Ros2Bridge(bridge_env.env)
    message = _command(names, values)
    for field, value in extra.items():
        setattr(message, field, value)
    bridge_env.node.pending.append(message)
    try:
        torch.testing.assert_close(bridge.advance(), torch.tensor([[3.0, 1.0, 2.0]]))
        assert len(bridge_env.node.warnings) == 1
    finally:
        bridge.close()


def test_position_commands_accept_and_ignore_valid_feedforward_arrays(bridge_env):
    bridge = Ros2Bridge(bridge_env.env)
    message = _command(["b", "a"], [8.0, 7.0])
    message.velocity = [100.0, -100.0]
    message.effort = [200.0, -200.0]
    bridge_env.node.pending.append(message)
    try:
        torch.testing.assert_close(bridge.advance(), torch.tensor([[3.0, 7.0, 8.0]]))
        assert not bridge_env.node.warnings
    finally:
        bridge.close()


@pytest.mark.parametrize("backend_owns_decimation", [False, True])
def test_publish_clock_tracks_integrated_time_across_pause_stop_resume_and_reset(bridge_env, backend_owns_decimation):
    bridge_env.env.cfg.ros2_bridge.frame_prefix = "sim/"
    bridge = Ros2Bridge(bridge_env.env)
    try:
        # Backend-owned decimation can integrate several physics ticks per sim.step().
        bridge_env.env.sim.backend_manager.handles_decimation = lambda: backend_owns_decimation
        bridge_env.env.sim.steps = 6173 if backend_owns_decimation else 24692
        bridge_env.env.sim.time = 12.346
        bridge.publish()
        state = bridge_env.node.published["/joint_states"][-1]
        assert state.name == ["b", "c", "a"]
        assert list(state.position) == [2.0, 3.0, 1.0]
        assert list(state.velocity) == pytest.approx([0.2, 0.3, 0.1])
        assert list(state.effort) == [20.0, 30.0, 10.0]
        clock = bridge_env.node.published["/clock"][-1].clock
        assert (clock.sec, clock.nanosec) == (12, 346000000)
        transform = bridge_env.node.published["/tf"][-1].transforms[0]
        assert transform.header.frame_id == "world" and transform.child_frame_id == "sim/tool"
        assert vars(transform.transform.translation) == dict(x=4.0, y=5.0, z=6.0)
        assert vars(transform.transform.rotation) == pytest.approx(dict(x=0.5, y=0.1, z=0.2, w=0.3))
        assert vars(transform.header.stamp) == vars(state.header.stamp) == vars(clock)
        # Paused and stopped headless loops may attempt steps without integrating.
        # Neither these attempts nor repeated publication may advance any timestamp.
        for _timeline_state in ("paused", "stopped"):
            bridge_env.env.sim.steps += 3
            bridge.publish()
            clock = bridge_env.node.published["/clock"][-1].clock
            state = bridge_env.node.published["/joint_states"][-1]
            transform = bridge_env.node.published["/tf"][-1].transforms[0]
            assert (clock.sec, clock.nanosec) == (12, 346000000)
            assert vars(transform.header.stamp) == vars(state.header.stamp) == vars(clock)
        # Resume one integration: backend-owned decimation spans four physics ticks.
        bridge_env.env.sim.steps += 1
        bridge_env.env.sim.time = 12.348 if backend_owns_decimation else 12.3465
        bridge.publish()
        clock = bridge_env.node.published["/clock"][-1].clock
        assert (clock.sec, clock.nanosec) == (12, 348000000 if backend_owns_decimation else 346500000)
        bridge_env.env.episode_length_buf = torch.zeros(1)
        bridge_env.asset.data.joint_pos.torch.zero_()
        bridge.publish()
        clock = bridge_env.node.published["/clock"][-1].clock
        assert (clock.sec, clock.nanosec) == (12, 348000000 if backend_owns_decimation else 346500000)
        bridge_env.env.sim.steps = 6200 if backend_owns_decimation else 24800
        bridge_env.env.sim.time = 12.4
        bridge.publish()
        clock = bridge_env.node.published["/clock"][-1].clock
        assert (clock.sec, clock.nanosec) == (12, 400000000)
    finally:
        bridge.close()


@pytest.mark.parametrize("period", [-1.0, float("nan"), float("inf"), -float("inf")])
def test_invalid_exchange_period_fails_before_creating_ros_node(bridge_env, period):
    bridge_env.env.cfg.ros2_bridge.exchange_period_s = period
    with pytest.raises(ValueError, match="exchange_period_s"):
        Ros2Bridge(bridge_env.env)
    assert not bridge_env.node.published


def test_exchange_cadence_holds_commands_and_publishes_only_on_anchored_grid(bridge_env, monkeypatch):
    period = 1.0 / 150.0
    bridge_env.env.cfg.ros2_bridge.exchange_period_s = period
    bridge = Ros2Bridge(bridge_env.env)
    gathers = []
    index_select = torch.index_select

    def gather(*args, **kwargs):
        gathers.append(bridge_env.env.sim.time)
        return index_select(*args, **kwargs)

    monkeypatch.setattr(torch, "index_select", gather)
    try:
        bridge.publish()  # Initial state is available without an advance or integration.
        assert bridge_env.node.spins == 0
        assert len(bridge_env.node.published["/clock"]) == 1
        bridge_env.node.pending.append(_command(["a"], [10.0]))
        actions = bridge._actions
        exchanges = []
        target = 1.0
        for step in range(2001):
            elapsed = step * 0.0005
            bridge_env.env.sim.time = elapsed
            due = elapsed >= (len(exchanges) + 1) * period
            before_gathers = len(gathers)
            assert bridge.advance() is actions
            if due:
                exchanges.append(step)
                target = 9.0 + len(exchanges)
            assert actions[0, 1].item() == target
            bridge_env.env.sim.time = (step + 1) * 0.0005
            bridge.publish()
            assert len(gathers) - before_gathers == (4 if due else 0)
            assert bridge_env.node.spins == 3 * len(exchanges)
            assert len(bridge_env.node.published["/clock"]) == len(exchanges) + 1
            if due:
                bridge_env.node.pending.append(_command(["a"], [target + 1.0]))
        assert len(exchanges) == 150
        assert exchanges[:3] == [14, 27, 40]
        assert {right - left for left, right in zip(exchanges, exchanges[1:])} == {13, 14}
        expected_stamps = [0] + [(step + 1) * 500000 for step in exchanges]
        for topic in ("/clock", "/joint_states", "/tf"):
            messages = bridge_env.node.published[topic]
            assert len(messages) == 151
            if topic == "/clock":
                stamps = [message.clock for message in messages]
            elif topic == "/tf":
                stamps = [message.transforms[0].header.stamp for message in messages]
            else:
                stamps = [message.header.stamp for message in messages]
            assert [stamp.sec * 1000000000 + stamp.nanosec for stamp in stamps] == expected_stamps
    finally:
        bridge.close()


def test_exchange_pause_resume_episode_reset_and_time_jump(bridge_env):
    bridge_env.env.cfg.ros2_bridge.exchange_period_s = 1.0 / 150.0
    bridge = Ros2Bridge(bridge_env.env)
    try:
        bridge.publish()
        bridge_env.node.pending.append(_command(["a"], [7.0]))
        for _ in range(10):  # Paused attempts neither poll ROS nor republish state.
            bridge.advance()
            bridge.publish()
        assert bridge_env.node.spins == 0
        assert len(bridge_env.node.published["/clock"]) == 1
        bridge_env.env.sim.time = 0.007
        assert bridge.advance()[0, 1].item() == 7.0
        bridge_env.env.sim.time = 0.0075
        bridge.publish()
        bridge_env.asset.data.joint_pos.torch.zero_()
        bridge_env.env.episode_length_buf = torch.zeros(1)
        bridge_env.env.sim.steps = 0
        bridge_env.node.pending.append(_command(["a"], [8.0]))
        for _ in range(10):
            assert bridge.advance()[0, 1].item() == 7.0
            bridge.publish()
        assert bridge_env.node.spins == 3
        bridge_env.env.sim.time = 0.041  # Cross several deadlines in one integration.
        assert bridge.advance()[0, 1].item() == 8.0
        bridge_env.env.sim.time = 0.0415
        bridge.publish()
        bridge.publish()
        assert len(bridge_env.node.published["/clock"]) == 3  # No catch-up burst.
        bridge_env.node.pending.append(_command(["a"], [9.0]))
        bridge_env.env.sim.time = 0.0465
        assert bridge.advance()[0, 1].item() == 8.0
        bridge.publish()
        bridge_env.env.sim.time = 0.047
        assert bridge.advance()[0, 1].item() == 9.0
        bridge_env.env.sim.time = 0.0475
        bridge.publish()
        assert bridge_env.node.spins == 9
        assert len(bridge_env.node.published["/clock"]) == 4
    finally:
        bridge.close()


def test_default_zero_period_keeps_every_step_exchange_even_while_paused(bridge_env):
    assert bridge_env.env.cfg.ros2_bridge.exchange_period_s == 0.0
    bridge = Ros2Bridge(bridge_env.env)
    try:
        bridge.publish()
        for value in (4.0, 5.0, 6.0):
            bridge_env.node.pending.append(_command(["a"], [value]))
            assert bridge.advance()[0, 1].item() == value
            bridge.publish()
        assert bridge_env.node.spins == 9
        assert len(bridge_env.node.published["/clock"]) == 4
        assert all(message.clock.nanosec == 0 for message in bridge_env.node.published["/clock"])
    finally:
        bridge.close()


def test_exchange_drains_arm_gripper_and_clock_callbacks_at_each_boundary(bridge_env):
    bridge_env.env.cfg.ros2_bridge.exchange_period_s = 1.0 / 150.0
    bridge = Ros2Bridge(bridge_env.env)
    clock_callbacks = []
    try:
        bridge.publish()
        for elapsed, value in ((0.007, 4.0), (0.0135, 8.0), (0.020, 12.0)):
            bridge_env.node.pending.extend(
                [
                    lambda: clock_callbacks.append(True),  # use_sim_time owns a subscription too.
                    _command(["unknown"], [value]),  # Rejection must not stop the drain.
                    _command(["a", "c"], [value, value + 1.0]),
                    _command(["b"], [value + 2.0]),
                    _command(["a"], [value + 3.0]),  # Last received target wins per joint.
                ]
            )
            bridge_env.env.sim.time = elapsed - 0.0005
            held = bridge._actions.clone()
            torch.testing.assert_close(bridge.advance(), held)
            assert len(bridge_env.node.pending) == 5
            bridge_env.env.sim.time = elapsed
            torch.testing.assert_close(bridge.advance(), torch.tensor([[value + 1.0, value + 3.0, value + 2.0]]))
            assert not bridge_env.node.pending
            bridge_env.env.sim.time = elapsed + 0.0005
            bridge.publish()
        assert len(clock_callbacks) == 3
        assert bridge_env.node.spins == 33  # Five callback/timeout pairs plus one empty pass each.
        assert len(bridge_env.node.warnings) == 3
        assert len(bridge_env.node.published["/clock"]) == 4
    finally:
        bridge.close()


@pytest.fixture
def transfer_copies(monkeypatch):
    """Observe explicit copy boundaries even when both devices are CPU."""
    copies = []
    copy_tensor = torch.Tensor.copy_

    def record(destination, source, *args, **kwargs):
        copies.append((destination, source))
        return copy_tensor(destination, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", record)
    return copies


def test_partial_messages_stage_one_command_copy_after_complete_ready_drain(bridge_env, transfer_copies):
    bridge = Ros2Bridge(bridge_env.env)
    transfer_copies.clear()  # Initialization seeds targets once from reset state.
    try:
        bridge_env.node.pending.extend(
            [
                _command(["a", "c"], [7.0, 9.0]),
                _command(["b"], [8.0]),
                _command(["unknown"], [99.0]),
                _command(["a"], [10.0]),
            ]
        )

        def observe_staging():
            assert not transfer_copies
            torch.testing.assert_close(bridge._actions, torch.tensor([[3.0, 1.0, 2.0]]))
            torch.testing.assert_close(bridge._command_buffer, torch.tensor([[9.0, 10.0, 8.0]]))

        bridge_env.node.pending.append(observe_staging)
        actions = bridge.advance()
        assert not bridge_env.node.pending
        assert len(transfer_copies) == 1
        assert transfer_copies[0][0] is actions
        assert transfer_copies[0][1] is bridge._command_buffer
        torch.testing.assert_close(actions, torch.tensor([[9.0, 10.0, 8.0]]))
        transfer_copies.clear()
        bridge.advance()  # No commands means no target transfer.
        bridge_env.node.pending.append(_command(["unknown"], [99.0]))
        bridge.advance()  # Rejected commands do not dirty staging.
        assert not transfer_copies
    finally:
        bridge.close()


def test_nonexchange_steps_do_not_transfer_commands_or_snapshot(bridge_env, transfer_copies, monkeypatch):
    bridge_env.env.cfg.ros2_bridge.exchange_period_s = 1.0 / 150.0
    bridge = Ros2Bridge(bridge_env.env)
    try:
        bridge.publish()
        transfer_copies.clear()
        bridge_env.node.pending.append(_command(["a"], [7.0]))

        def unexpected_gather(*args, **kwargs):
            pytest.fail("Nonexchange step gathered state.")

        monkeypatch.setattr(torch, "index_select", unexpected_gather)
        for step in range(14):
            bridge_env.env.sim.time = step * 0.0005
            assert bridge.advance() is bridge._actions
            bridge.publish()
        assert not transfer_copies
        assert bridge_env.node.spins == 0
        assert len(bridge_env.node.pending) == 1
        assert len(bridge_env.node.published["/clock"]) == 1
    finally:
        bridge.close()


@pytest.mark.parametrize("bodies", [(), ("tool",), ("apple", "tool")])
def test_joint_state_and_only_configured_tf_share_one_reused_snapshot(bridge_env, transfer_copies, bodies):
    bridge_env.env.cfg.ros2_bridge.body_names = bodies
    bridge = Ros2Bridge(bridge_env.env)
    transfer_copies.clear()
    try:
        gather = bridge._gather_buffer
        snapshot = bridge._snapshot_buffer
        command = bridge._command_buffer
        assert gather.numel() == snapshot.numel() == 9 + 7 * len(bodies)
        for view in (bridge._state_buffer, bridge._pose_buffer):
            assert view.untyped_storage().data_ptr() == snapshot.untyped_storage().data_ptr()
        for view in (bridge._joint_gather, bridge._pose_gather):
            assert view.untyped_storage().data_ptr() == gather.untyped_storage().data_ptr()
        for iteration in range(2):
            bridge_env.asset.data.joint_pos.torch[0, 1] = 2.0 + iteration
            bridge.publish()
            assert len(transfer_copies) == iteration + 1
            assert transfer_copies[-1][0] is snapshot
            assert transfer_copies[-1][1] is gather
            assert bridge._gather_buffer is gather
            assert bridge._snapshot_buffer is snapshot
            assert bridge._command_buffer is command
            state = bridge_env.node.published["/joint_states"][-1]
            assert state.position == [2.0 + iteration, 3.0, 1.0]
            assert state.velocity == pytest.approx([0.2, 0.3, 0.1])
            assert state.effort == [20.0, 30.0, 10.0]
            if bodies:
                transforms = bridge_env.node.published["/tf"][-1].transforms
                assert [transform.child_frame_id for transform in transforms] == list(bodies)
                assert all(transform.child_frame_id != "plant" for transform in transforms)
                assert [transform.transform.translation.x for transform in transforms] == [
                    200.0 if name == "apple" else 4.0 for name in bodies
                ]
            else:
                assert not bridge_env.node.published["/tf"]
    finally:
        bridge.close()


@pytest.mark.parametrize("invalid", ["multiple_envs", "offset", "scale", "clip", "unselected", "duplicate_joint"])
def test_unsupported_action_contract_fails_before_creating_ros_node(bridge_env, invalid):
    if invalid == "multiple_envs":
        bridge_env.env.num_envs = 2
    elif invalid == "offset":
        bridge_env.terms["arm"].cfg.use_default_offset = True
    elif invalid == "scale":
        bridge_env.terms["arm"]._scale = torch.tensor([[1.0, 2.0]])
    elif invalid == "clip":
        bridge_env.terms["arm"].cfg.clip = {".*": (-1.0, 1.0)}
    elif invalid == "unselected":
        bridge_env.terms["mechanics"].action_dim = 1
    else:
        bridge_env.terms["gripper"]._joint_ids = torch.tensor([0])
    with pytest.raises(ValueError):
        Ros2Bridge(bridge_env.env)
    assert not bridge_env.node.published


def test_bridge_configuration_supports_native_copy_and_nested_overrides():
    """Task config construction and Hydra-style nested updates must remain mutable."""
    from isaaclab.utils import configclass

    @configclass
    class TaskCfg:
        ros2_bridge: Ros2BridgeCfg = Ros2BridgeCfg("robot", ("a",), ("arm",))

    cfg = TaskCfg().copy()
    cfg.from_dict({"ros2_bridge": {"joint_command_topic": "/custom_commands"}})
    assert cfg.ros2_bridge.joint_command_topic == "/custom_commands"
    assert TaskCfg().ros2_bridge.joint_command_topic == "/joint_commands"


@pytest.mark.parametrize("period", [0.0, 1.0 / 150.0])
def test_ros2_real_private_context_transport(bridge_model, period):
    """Exercise actual Jazzy executors and message serialization without a simulator."""
    import time
    import uuid

    rclpy = pytest.importorskip("rclpy")
    from rclpy.context import Context
    from rclpy.duration import Duration
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import JointState as RosJointState

    prefix = f"/isaaclab_test_{uuid.uuid4().hex}"
    bridge_model.env.cfg.ros2_bridge.joint_state_topic = prefix + "/state"
    bridge_model.env.cfg.ros2_bridge.joint_command_topic = prefix + "/command"
    bridge_model.env.cfg.ros2_bridge.exchange_period_s = period
    bridge_model.env.sim.steps = 250
    bridge_model.env.sim.time = 0.125
    context = Context()
    rclpy.init(context=context)
    peer = rclpy.create_node("isaaclab_bridge_test_peer", context=context)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(peer)
    states = []
    subscription = peer.create_subscription(RosJointState, prefix + "/state", states.append, 10)
    publisher = peer.create_publisher(RosJointState, prefix + "/command", 10)
    gripper_publisher = peer.create_publisher(RosJointState, prefix + "/command", 10)
    bridge = None
    try:
        bridge = Ros2Bridge(bridge_model.env)
        assert bridge._node.get_parameter("use_sim_time").value is True
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if (
                publisher.get_subscription_count()
                and gripper_publisher.get_subscription_count()
                and bridge._state_pub.get_subscription_count()
                and bridge._clock_pub.get_subscription_count()
            ):
                break
            executor.spin_once(timeout_sec=0.01)
        else:
            pytest.fail("Private-context ROS endpoints did not discover each other.")
        bridge.publish()  # Queue the bridge's own use_sim_time clock callback too.
        assert bridge._clock_pub.wait_for_all_acked(Duration(seconds=5.0))
        malformed = RosJointState()
        malformed.name = ["unknown"]
        malformed.position = [99.0]
        publisher.publish(malformed)
        command = RosJointState()
        command.name = ["a", "c"]
        command.position = [7.0, 9.0]
        command.velocity = [2.0, 3.0]
        command.effort = [3.0, 4.0]
        publisher.publish(command)
        gripper = RosJointState()
        gripper.name = ["b"]
        gripper.position = [8.0]
        gripper_publisher.publish(gripper)
        assert publisher.wait_for_all_acked(Duration(seconds=5.0))
        assert gripper_publisher.wait_for_all_acked(Duration(seconds=5.0))
        if period:
            torch.testing.assert_close(bridge.advance(), torch.tensor([[3.0, 1.0, 2.0]]))
        bridge_model.env.sim.time = 0.132
        actions = bridge.advance()  # Exactly one exchange must consume every queued command.
        torch.testing.assert_close(actions, torch.tensor([[9.0, 7.0, 8.0]]))
        bridge_model.env.sim.time = 0.1325
        bridge.publish()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.01)
            if states and states[-1].header.stamp.nanosec == 132500000:
                break
        else:
            pytest.fail("Private-context ROS bridge did not publish the exchanged state.")
        assert list(states[-1].position) == [2.0, 3.0, 1.0]
        assert list(states[-1].effort) == [20.0, 30.0, 10.0]
        stamp = states[-1].header.stamp
        assert (stamp.sec, stamp.nanosec) == (0, 132500000)
        assert subscription is not None
    finally:
        if bridge is not None:
            bridge.close()
            bridge.close()
        executor.shutdown()
        peer.destroy_node()
        context.try_shutdown()
