ClinicalTrialEligibilityChecker
==================

ClinicalTrialEligibilityChecker is an end-to-end clinical trial eligibility assessment
system designed to support both automated relevance scoring and
clinician-facing interpretability.

The system integrates large language models, neural ranking models,
and a web-based user interface to enable transparent, reproducible,
and scalable patient–trial matching.


SYSTEM OVERVIEW
---------------
ClinicalTrialEligibilityChecker consists of two main components:

1. ClinicalTrialEligibilityChecker_Server
   A FastAPI-based inference server that exposes REST APIs for
   eligibility scoring, explanation generation, and batch evaluation.

2. ClinicalTrialEligibilityChecker_Client
   A Streamlit-based client application that allows clinicians
   and researchers to interact with the system through a
   user-friendly interface.


ARCHITECTURE
------------

Client (Streamlit UI)
    |
    | REST API calls
    v
Server (FastAPI)
    |
    | Model inference
    v
NEUREQ and TCH_CLF Models
    |
    | Structured audit logs
    v
Persistent JSON Logs

The client and server are intentionally decoupled to allow:
- Independent deployment
- Easier auditing and debugging
- Future replacement or extension of models or UI components


MODELS
------

NEUREQ
------
- Question-driven eligibility modeling framework.
- Uses a fixed 10-question schema capturing core eligibility dimensions.
- Each question produces a ternary response (YES / NO / NA) with justification.
- Responses are scored using a trained neural model.

TCH_CLF
-------
- Teacher Longformer-based clinical trial reranker.
- Extracts structured trial attributes (age, gender, conditions, criteria).
- Produces a relevance score with optional deterministic reasoning.


BATCH EVALUATION
----------------
The system supports batch evaluation where:
- Multiple patient cases are evaluated against multiple trials.
- Execution is sequential and GPU-safe.
- Live progress is exposed to the client.
- Partial results are available before batch completion.
- All explanations are served from cached audit logs.


AUDITABILITY AND REPRODUCIBILITY
--------------------------------
ClinicalTrialEligibilityChecker emphasizes clinical transparency by:
- Logging all intermediate representations
- Storing deterministic seeds for reproducibility
- Persisting per-trial explanations
- Separating inference from presentation

All logs are stored as structured JSON files and can be inspected
independently of the client application.


DIRECTORY STRUCTURE
-------------------

ClinicalTrialEligibilityChecker/
├── ClinicalTrialEligibilityChecker_Server/
│   ├── server.py
│   ├── models/
│   ├── audit_logs/
│   └── README.txt
│
├── ClinicalTrialEligibilityChecker_Client/
│   ├── app.py
│   ├── assets/
│   ├── config.json
│   └── README.txt
│
└── README.txt   (this file)


SETUP AND EXECUTION
-------------------
Refer to the README files inside each subdirectory for
component-specific setup and execution instructions:

- ClinicalTrialEligibilityChecker_Server/README.txt
- ClinicalTrialEligibilityChecker_Client/README.txt


INTENDED USE
------------
This system is intended for:
- Clinical trial eligibility research
- Model evaluation and benchmarking
- Decision-support prototyping
- Human-in-the-loop eligibility inspection

It is not intended to replace clinical judgment.


LICENSE
-------
Internal research use only.
Contact the authors for redistribution, clinical deployment,
or commercial use permissions.
