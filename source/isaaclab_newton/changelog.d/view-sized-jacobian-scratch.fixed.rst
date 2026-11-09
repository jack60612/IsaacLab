* Fixed Newton articulation Jacobian and mass-matrix queries allocating (and zeroing on every read) a scratch
  sized ``model.articulation_count x 6 * model.max_joints_per_articulation x model.max_dofs_per_articulation``.
  In scenes where a small robot shares the model with a large articulation and many free bodies in every world,
  this grew with the whole scene (e.g. 3.7 GB per world for a 12-DoF arm). The scratch now uses Newton's
  view-layout evaluation and is sized by the asset's own articulation.
