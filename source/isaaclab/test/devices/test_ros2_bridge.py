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
    node = SimpleNamespace(published={}, pending=[], warnings=[], destroyed=False)

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

    class Executor:
        def __init__(self, *, context):
            assert context is bridge_context
            self.node = None
            self.stopped = False

        def add_node(self, value):
            self.node = value

        def spin_once(self, *, timeout_sec):
            assert self.node is node and timeout_sec == 0.0
            if node.pending:
                node.callback(node.pending.pop(0))

        def shutdown(self):
            self.stopped = True

    bridge_context = context

    modules = {
        "rclpy": dict(init=lambda **kwargs: None, create_node=lambda *args, **kwargs: node),
        "rclpy.context": dict(Context=lambda: context),
        "rclpy.executors": dict(SingleThreadedExecutor=Executor),
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
        body_link_pose_w=SimpleNamespace(torch=torch.tensor([[[4.0, 5.0, 6.0, 0.5, 0.1, 0.2, 0.3]]])),
    )
    asset = SimpleNamespace(joint_names=["a", "b", "c"], body_names=["tool"], num_joints=3, data=data)
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
    env.sim.physics_manager = SimpleNamespace(
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
        bridge_env.env.sim.physics_manager.handles_decimation = lambda: backend_owns_decimation
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


def test_ros2_real_private_context_transport(bridge_model):
    """Exercise actual Jazzy executors and message serialization without a simulator."""
    import time
    import uuid

    rclpy = pytest.importorskip("rclpy")
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import JointState as RosJointState

    prefix = f"/isaaclab_test_{uuid.uuid4().hex}"
    bridge_model.env.cfg.ros2_bridge.joint_state_topic = prefix + "/state"
    bridge_model.env.cfg.ros2_bridge.joint_command_topic = prefix + "/command"
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
    bridge = None
    try:
        bridge = Ros2Bridge(bridge_model.env)
        assert bridge._node.get_parameter("use_sim_time").value is True
        command = RosJointState()
        command.name = ["a"]
        command.position = [7.0]
        command.velocity = [2.0]
        command.effort = [3.0]
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            publisher.publish(command)
            actions = bridge.advance()
            bridge.publish()
            executor.spin_once(timeout_sec=0.01)
            if states and actions[0, 1].item() == 7.0:
                break
        else:
            pytest.fail("Private-context ROS bridge did not exchange real JointState messages.")
        torch.testing.assert_close(actions, torch.tensor([[3.0, 7.0, 2.0]]))
        assert list(states[-1].position) == [2.0, 3.0, 1.0]
        assert list(states[-1].effort) == [20.0, 30.0, 10.0]
        stamp = states[-1].header.stamp
        assert (stamp.sec, stamp.nanosec) == (0, 125000000)
        assert subscription is not None
    finally:
        if bridge is not None:
            bridge.close()
            bridge.close()
        executor.shutdown()
        peer.destroy_node()
        context.try_shutdown()
