* Fixed Newton fixed-base articulation root pose writers updating only a copy of the root pose when the articulation
  has interior fixed joints. The root pose is now bound directly to the root-joint rows of ``model.joint_X_p``, so
  writes move the model, the solver and the links, and ``root_link_pose_w`` reads back the simulated pose.
