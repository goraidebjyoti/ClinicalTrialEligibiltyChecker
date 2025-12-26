# ClinicalTrialEligibilityChecker

ClinicalTrialEligibilityChecker is an end-to-end clinical trial eligibility
assessment system designed to support both automated relevance scoring and
clinician-facing interpretability.

The system integrates large language models, neural ranking models,
and a web-based user interface to enable transparent, reproducible,
and scalable patient–trial matching.

## System Overview

ClinicalTrialEligibilityChecker consists of two main components:

1. **ClinicalTrialEligibilityChecker_Server**  
   A FastAPI-based inference server that exposes REST APIs for
   eligibility scoring, explanation generation, and batch evaluation.

2. **ClinicalTrialEligibilityChecker_Client**  
   A Streamlit-based client application that allows clinicians
   and researchers to interact with the system through a
   user-friendly interface.

## Architecture

The system follows a clean client–server architecture:

- **Client (Streamlit UI)**  
  Handles user interaction, visualization, and clinician-facing workflows.

- **Server (FastAPI)**  
  Exposes REST APIs and orchestrates model inference.

- **Model Layer**  
  - NEUREQ neural eligibility model  
  - TCH_CLF teacher Longformer reranker  

- **Audit Layer**  
  Structured JSON logs persisted for traceability and inspection.

The client and server are intentionally decoupled to allow:
- Independent deployment
- Easier auditing and debugging
- Future replacement or extension of models or UI components

## Models

### NEUREQ
- Question-driven eligibility modeling framework.
- Uses a fixed 10-question schema capturing core eligibility dimensions.
- Each question produces a ternary response (YES / NO / NA) with justification.
- Responses are scored using a trained neural model.

### TCH_CLF
- Teacher Longformer-based clinical trial reranker.
- Extracts structured trial attributes (age, gender, conditions, criteria).
- Produces a relevance score with optional deterministic reasoning.

## Model Checkpoints

Trained model checkpoint files are not included in this repository due to
size constraints.

The inference server expects the required `.pt` files to be present under:

ClinicalTrialEligibilityChecker_Server/models/

Instructions and download links for the official model checkpoints are
provided in:

ClinicalTrialEligibilityChecker_Server/models/readme.md


## Batch Evaluation

The system supports batch evaluation where:
- Multiple patient cases are evaluated against multiple trials.
- Execution is sequential and GPU-safe.
- Live progress is exposed to the client.
- Partial results are available before batch completion.
- All explanations are served from cached audit logs.

## Auditability and Reproducibility

ClinicalTrialEligibilityChecker emphasizes clinical transparency by:
- Logging all intermediate representations
- Storing deterministic seeds for reproducibility
- Persisting per-trial explanations
- Separating inference from presentation

All logs are stored as structured JSON files and can be inspected
independently of the client application.

## Directory Structure

```text
ClinicalTrialEligibilityChecker/
├── ClinicalTrialEligibilityChecker_Server/
│   ├── server.py
│   ├── models/
│   ├── audit_logs/
│   └── readme.md
│
├── ClinicalTrialEligibilityChecker_Client/
│   ├── app.py
│   ├── assets/
│   ├── config.json
│   └── readme.md
│
└── readme.md
```

## Setup and Execution

Refer to the README files inside each subdirectory for
component-specific setup and execution instructions:

- ClinicalTrialEligibilityChecker_Server/readme.md
- ClinicalTrialEligibilityChecker_Client/readme.md

## Intended Use

This system is intended for:
- Clinical trial eligibility research
- Model evaluation and benchmarking
- Decision-support prototyping
- Human-in-the-loop eligibility inspection

It is not intended to replace clinical judgment.

## License

Internal research use only.  
Contact the authors for redistribution, clinical deployment,
or commercial use permissions.

