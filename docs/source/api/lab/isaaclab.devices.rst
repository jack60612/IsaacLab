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
Named partial commands update only those position targets.
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
The ``wp.transformf`` Torch view already stores quaternion components in ROS-compatible
xyzw order. All messages share the monotonic simulation step count times physics dt,
independent of episode resets. When the backend's ``handles_decimation()`` is true,
each counted step spans ``env.cfg.decimation`` physics ticks; the clock includes this factor.
This clock contract assumes fixed physics dt and decimation ownership for the run.
Commands are processed just before ``env.step`` and state is published afterward,
as well as once after the initial reset. The bridge owns a private ROS context and
a single-threaded executor bound to that context; both close before the environment.

.. autoclass:: isaaclab.devices.ros2.Ros2BridgeCfg
   :members:

.. autoclass:: isaaclab.devices.ros2.Ros2Bridge
   :members:
