* :meth:`~isaaclab_newton.physics.NewtonManager.set_decimation` keeps the full-loop CUDA graph of every decimation captured since the last reset and replays it when that decimation returns, instead of recapturing on every call.
* Simulation time advances by the configured physics dt once per physics step, as on PhysX, so it no longer depends on how many physics steps one call folds together.
