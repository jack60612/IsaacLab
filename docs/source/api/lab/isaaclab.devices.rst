isaaclab.devices
================

.. automodule:: isaaclab.devices

  .. rubric:: Classes

  .. autosummary::

    DeviceBase
    RetargeterBase
    Se2Gamepad
    Se3Gamepad
    Se2Keyboard
    Se3Keyboard
    Se2SpaceMouse
    Se3SpaceMouse
    HaplyDevice
    OpenXRDevice
    ManusVive
    openxr.retargeters.GripperRetargeter
    openxr.retargeters.Se3AbsRetargeter
    openxr.retargeters.Se3RelRetargeter
    openxr.retargeters.GR1T2Retargeter

  .. rubric:: Modules

  .. autosummary::

    openxr.retargeters

Device Base
-----------

.. autoclass:: DeviceBase
    :members:

Retargeter Base
---------------

.. autoclass:: RetargeterBase
    :members:

Game Pad
--------

.. autoclass:: Se2Gamepad
    :members:
    :inherited-members:
    :show-inheritance:
    :noindex:

.. autoclass:: Se3Gamepad
    :members:
    :inherited-members:
    :show-inheritance:
    :noindex:

Keyboard
--------

.. autoclass:: Se2Keyboard
    :members:
    :inherited-members:
    :show-inheritance:
    :noindex:

.. autoclass:: Se3Keyboard
    :members:
    :inherited-members:
    :show-inheritance:
    :noindex:

Space Mouse
-----------

.. autoclass:: Se2SpaceMouse
    :members:
    :inherited-members:
    :show-inheritance:
    :noindex:

.. autoclass:: Se3SpaceMouse
    :members:
    :inherited-members:
    :show-inheritance:
    :noindex:

Haply
-----

.. autoclass:: HaplyDevice
    :members:
    :inherited-members:
    :show-inheritance:
    :noindex:

OpenXR
------

.. autoclass:: OpenXRDevice
    :members:
    :inherited-members:
    :show-inheritance:
    :noindex:

Manus + Vive
------------

.. autoclass:: ManusVive
    :members:
    :inherited-members:
    :show-inheritance:
    :noindex:

Retargeters
-----------

.. autoclass:: isaaclab.devices.openxr.retargeters.GripperRetargeter
    :members:
    :inherited-members:
    :show-inheritance:
    :noindex:

.. autoclass:: isaaclab.devices.openxr.retargeters.Se3AbsRetargeter
    :members:
    :inherited-members:
    :show-inheritance:
    :noindex:

.. autoclass:: isaaclab.devices.openxr.retargeters.Se3RelRetargeter
    :members:
    :inherited-members:
    :show-inheritance:
    :noindex:

.. autoclass:: isaaclab.devices.openxr.retargeters.GR1T2Retargeter
    :members:
    :inherited-members:
    :show-inheritance:
    :noindex:

Additional Public Classes
-------------------------

The following classes are part of the public :mod:`isaaclab.devices` API.

.. currentmodule:: isaaclab.devices

.. autosummary::
   :nosignatures:

   DeviceCfg
   DevicesCfg
   HaplyDeviceCfg
   RetargeterCfg
   Se2GamepadCfg
   Se2KeyboardCfg
   Se2SpaceMouseCfg
   Se3GamepadCfg
   Se3KeyboardCfg
   Se3SpaceMouseCfg

.. autoclass:: DeviceCfg
   :show-inheritance:

.. autoclass:: DevicesCfg
   :show-inheritance:

.. autoclass:: HaplyDeviceCfg
   :show-inheritance:

.. autoclass:: RetargeterCfg
   :show-inheritance:

.. autoclass:: Se2GamepadCfg
   :show-inheritance:

.. autoclass:: Se2KeyboardCfg
   :show-inheritance:

.. autoclass:: Se2SpaceMouseCfg
   :show-inheritance:

.. autoclass:: Se3GamepadCfg
   :show-inheritance:

.. autoclass:: Se3KeyboardCfg
   :show-inheritance:

.. autoclass:: Se3SpaceMouseCfg
   :show-inheritance:

Optional ROS 2 Joint Bridge
--------------------------

The existing zero agent supports ``--ros2`` for a single manager-based environment.
It uses :class:`isaaclab.devices.ros2.Ros2Bridge` instead of the zero policy; there is
no second simulation loop or wall-clock timer. Without the flag, ROS is not imported.
ROS dependencies are external: source a ROS 2 Jazzy installation and use Python 3.12.
If your task setup script clears ROS environment variables, source it **before** ROS.

.. code-block:: bash

   source /opt/ros/jazzy/setup.bash
   uv run python scripts/environments/zero_agent.py --task YOUR_REGISTERED_TASK --num_envs 1 --ros2

When using a separate simulator Python environment, ensure this checkout's source
packages take precedence over any other editable Isaac Lab installation:

.. code-block:: bash

   export PYTHONPATH="$PWD/source/isaaclab:$PWD/source/isaaclab_rl:$PWD/source/isaaclab_tasks:$PYTHONPATH"

The task must expose ``env.cfg.ros2_bridge`` as a
:class:`isaaclab.devices.ros2.Ros2BridgeCfg`, for example:

.. code-block:: python

   from isaaclab.devices.ros2 import Ros2BridgeCfg

   ros2_bridge = Ros2BridgeCfg(
       articulation_name="robot",
       joint_names=("joint_a", "joint_b"),
       action_terms=("arm",),
       body_names=("tool_link",),
   )

Selected action terms must be absolute ``JointPositionAction`` terms on that
articulation, with ``scale=1.0``, ``offset=0.0``, ``use_default_offset=False`` and
no clipping. Their resolved joint order, rather than the ROS message order, determines
action-vector indices. Every configured joint must be controlled exactly once.
Other action terms are allowed only if they have zero action width.

Publish ``sensor_msgs/msg/JointState`` on ``/joint_commands`` (configurable via
``joint_command_topic``). Commands contain unique joint names and matching finite
absolute positions in radians (meters for prismatic joints). Optional velocity and
effort arrays must be empty or match the name count and contain finite values.
These feedforward fields are validated but ignored by the position-action bridge,
allowing stock topic-based ROS control publishers to send full JointState commands.
Named partial commands update only those position targets. At each exchange the
executor drains all ready callbacks, so separate arm and gripper publishers are
processed before the same physics step; the last received value wins per joint.
Unspecified joints hold their previous target, initialized from the state after
``env.reset()``. Targets deliberately persist across automatic episode resets;
they are not reinterpreted as offsets or reset to zero. Malformed, duplicate,
unknown-joint or nonfinite commands are rejected atomically with a warning.

The bridge publishes measured position, velocity and **applied torque** on
``/joint_states`` (configurable via ``joint_state_topic``), ``rosgraph_msgs/msg/Clock``
on ``/clock``, and ``tf2_msgs/msg/TFMessage`` containing ``geometry_msgs/msg/TransformStamped``
on ``/tf`` for configured bodies. Transforms map ``world_frame`` (default ``world``)
to ``frame_prefix + body_name`` (the prefix defaults to empty), using world link poses.
Set ``frame_prefix="sim/"`` when a robot state publisher already owns the URDF
frame tree, avoiding multiple parents for the same TF child frame.
``body_names`` is an exact, ordered subset of the selected articulation's bodies;
unconfigured bodies and other scene assets (including plants and apples) are never
gathered or published. An empty tuple disables body-pose gathering and TF publication.
The ``wp.transformf`` Torch view already stores quaternion components in ROS-compatible
xyzw order. All messages share the backend's integrated simulation time from
``physics_manager.get_simulation_time()``, independent of episode resets.
Paused or stopped step attempts that do not integrate physics leave the timestamps
unchanged. Backend-owned decimation is already included in this time; the bridge
does not multiply it again or use wall-clock time.
The host calls ``advance()`` before every ``env.step()`` and ``publish()`` afterward,
as well as publishing once after the initial reset. Set ``exchange_period_s`` to a
finite, nonnegative interval in seconds to limit ROS work without changing the
environment's physics timestep. The default, zero, preserves an exchange on every
environment step. With a positive period, the initial state is published immediately;
subsequent command processing occurs at the first pre-step integrated-time sample
meeting each deadline, followed by one post-step publication of joint state, TF and
clock. ``advance()`` also ends each environment step on the first physics step whose
integrated time reaches the next deadline, using
:meth:`~isaaclab.envs.ManagerBasedEnv.set_step_decimation` with at most
``cfg.decimation`` physics steps. Exchanges therefore occur on the same physics steps,
at bit-identical integrated times, as with a decimation of one, without an environment
step per physics step; with ``cfg.decimation`` at least the longest deadline interval,
each environment step spans one exchange interval and its publication carries the
next deadline's time. Between exchanges, the same action tensor holds its targets,
without polling ROS, gathering state or copying it to the CPU.
Ready callbacks update a persistent CPU command staging buffer. After the complete
ready drain (including Jazzy's productive-readiness-pass retry), one blocking copy
updates the device action tensor if any valid command arrived. Multiple partial
messages therefore share one transfer, with the last received value winning per
joint; invalid-only or empty drains perform none. Blocking the copy ensures the
staging buffer is safe to reuse on the next exchange.
Publication gathers the configured joint positions, velocities, applied torques
and body poses into one contiguous device buffer, then copies that entire snapshot
to the CPU once for joint-state and TF serialization. Buffers and their joint/pose
views are allocated once during construction, including on CPU-only environments.

Deadlines are anchored to the initial publication time, not accumulated from the
last exchange. For example, a period of ``1.0 / 150.0`` with 1 ms physics steps
first reaches its deadlines on physics steps 7, 14, 20, 27, 34 and 40, alternating
7, 7 and 6 physics steps instead of rounding to a fixed step count. With
``cfg.decimation = 1``, the first command exchange is at 7 ms and its post-step
publication at 8 ms; with ``cfg.decimation = 7``, that publication is at 14 ms, the
next exchange. Messages carry their actual integrated time, not the ideal deadline.
Paused/stopped attempts perform no exchanges once the
initial state is published until integrated time meets the next deadline.
Episode resets do not restart the cadence. A larger integration jump produces one
exchange and skips missed deadlines, never a burst of stale publications.
The bridge owns a private ROS context and a single-threaded executor bound to that
context; both close before the environment.

.. autoclass:: isaaclab.devices.ros2.Ros2BridgeCfg
   :members:

.. autoclass:: isaaclab.devices.ros2.Ros2Bridge
   :members:
