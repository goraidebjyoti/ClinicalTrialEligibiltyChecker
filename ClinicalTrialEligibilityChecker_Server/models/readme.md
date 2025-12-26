Models Directory
================

This directory contains all trained model checkpoints used by the
EligibilityChecker inference server.

The models stored here are loaded by the server at runtime and are
required for both single and batch eligibility evaluation.


CONTENTS
--------

neureq_best.pt
--------------
- Trained neural eligibility model used by the NEUREQ pipeline.
- Implements a lightweight LSTM-based classifier.
- Operates on a fixed-length vector derived from the 10-question
  eligibility schema.
- Outputs a scalar eligibility score in the range [0, 1].

best_teacher_alpha0.2.pt
------------------------
- Trained teacher reranker model used by the TCH_CLF pipeline.
- Built on top of a Clinical Longformer encoder.
- Produces a relevance score between a patient case and a clinical trial.
- Designed to align with the offline teacher model used during training.


MODEL USAGE
-----------
- Models are loaded by the server during startup or on first use.
- NEUREQ model is always loaded at startup (CPU-safe and lightweight).
- TCH_CLF model is lazy-loaded to conserve memory and GPU resources.
- Model paths are defined in server.py and can be updated as needed.


COMPATIBILITY
-------------
- The NEUREQ model expects a 10 × 1 input tensor corresponding to
  ternary eligibility responses.
- The TCH_CLF model expects tokenized input compatible with the
  Clinical Longformer tokenizer.
- Model checkpoints are not interchangeable across architectures.


REPRODUCIBILITY
---------------
- These checkpoints correspond to fixed training runs.
- Inference is deterministic when used with fixed random seeds.
- Any change to these files may alter scoring behavior and should be
  documented.


NOTES
-----
- Do not rename model files without updating paths in server.py.
- Do not modify model files unless retraining is intended.
- For archival or versioned experiments, copy this directory rather
  than overwriting existing checkpoints.


LICENSE
-------
Internal research artifacts.
Not for redistribution without permission.
