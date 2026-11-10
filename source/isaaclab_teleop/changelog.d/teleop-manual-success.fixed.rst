* Fixed ``teleop_se3_agent`` ending like a successful session when the environment could not be created; the
  error now propagates and the command exits non-zero.
* Fixed ``teleop_se3_agent`` resetting the environment inside ``env.step()`` on success, which hid the successful
  state before it was rendered. Success is now evaluated manually, as in ``record_demos``, reported once per
  episode, and the operator's reset starts the next episode.
