* Added an optional lockstep ROS 2 joint-position bridge with named partial commands, measured joint states, simulation clock, and configured world-frame body transforms.
* The ROS 2 bridge timestamps messages with backend-integrated simulation time, including backend-owned decimation, so paused or stopped attempts that do not integrate physics cannot advance the clock.
