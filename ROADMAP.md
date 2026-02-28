# AID Thesis Roadmap

This file is the authoritative high-level roadmap for the repository.
Detailed task definitions remain under `src/cnn/ai/active_tasks/` and are not deleted.

## Project Status Summary

- Research infrastructure is complete and stable.
- Governance and reproducibility are enforced.
- Robustness methodology upgrade is the current focus.
- Multi-model experiments are deferred intentionally to protect thesis scope and delivery stability.

## Completed

- `0007` Evaluation Upgrade and Calibration
  Status: Implemented in code and verified through the Docker evaluation flow, including ROC-AUC, PR-AUC, thresholded metrics, calibration metrics, and report generation.
- `0022` Model Selection Governance and Test Access Policy
  Status: Implemented in code and verified, including immutable baseline hashing, access logging, drift blocking, and invalidation of override runs for thesis claims.

## Infrastructure Completed (Method Pending)

- `0001` Audit Current Pipeline
  Status: The current pipeline has been audited and documented, but the audit findings still need to drive the remaining method changes.
- `0003` Dataset Leakage and Split Fix
  Status: Deterministic split manifests, explicit label mapping, and robust manifest loading exist, but dataset-level leakage analysis is not yet complete at thesis depth.
- `0008` Real-World Holdout Protocol
  Status: Governance and evaluation hooks can support held-out splits, but the real-world holdout definition and curation are not yet complete.
- `0010` Experiment Tracking System
  Status: Reports, access logs, hashes, and governance artifacts exist, but a full experiment registry and run taxonomy are still incomplete.
- `0019` Reproducibility and Paper Artifacts
  Status: Reproducibility enforcement is partially implemented through immutable governance, but final paper artifacts and reproduction packaging are not finished.
- `0020` Dataset Provenance, Metadata, and Version Manifest
  Status: Manifest schema supports `dataset_id` and `domain`, but complete provenance/version governance is not yet fully populated.
- `0021` Cross-Dataset and Leave-One-Domain-Out Generalization
  Status: Subset filtering by `dataset_id` and `domain` exists, but full cross-dataset evaluation campaigns have not yet been executed.

## In Progress

- `0002` Audit DRCT
  Status: Still part of the active technical investigation and not yet translated into an implemented benchmark or comparison result.
- `0004` Robust Augmentation Design
  Status: Basic augmentation hooks exist in the training loop, but a hypothesis-driven augmentation policy is not yet finalized.
- `0005` Regularization Upgrade
  Status: The baseline CNN already uses dropout, batch normalization, learning-rate decay, and early stopping, but a controlled regularization study is not complete.
- `0006` Consistency Training Module
  Status: Planned as the next robustness-method upgrade, but not yet implemented in the training objective.
- `0011` Agentic Workflow Design
  Status: Task scaffolding and local workflow docs exist, but the operational research workflow is still evolving.
- `0015` Dataset Rebalancing Strategy
  Status: The manifest and evaluation layers can support this work, but no rebalancing protocol is implemented yet.
- `0016` Baselines and Benchmark Suite
  Status: The evaluation stack is ready, but the benchmark suite itself is still being assembled and run.
- `0024` Shortcut and Spurious Signal Audit
  Status: Recognized as critical, but the audit remains a planned investigation rather than an implemented analysis pipeline.
- `0026` Problem Formulation and Claim Scope
  Status: The framing is documented in tasks, but final thesis claim boundaries are still being tightened alongside the method plan.

## Deferred (Out of Scope for Final Delivery)

- `0009` Multi-Model Architecture
  Deferred because a single-model thesis with strong governance is the current scope boundary, and multi-model work would expand implementation and tuning cost significantly.
- `0012` Model Ensemble Experiment
  Deferred because ensemble gains are secondary to establishing a defensible single-model robustness baseline within the thesis timeline.
- `0013` Frequency-Domain Experiment
  Deferred because it adds a new modeling branch and comparison surface that is not necessary for the final controlled delivery target.
- `0014` Failure Analysis Pipeline
  Deferred because full error-taxonomy automation would broaden tooling scope beyond the minimum thesis delivery, even though manual analysis remains important.
- `0017` Ablation Matrix
  Deferred because a large ablation campaign would require a broader experiment budget than the final delivery scope allows.
- `0018` Statistical Rigor and CI
  Deferred because the full multi-seed significance workflow is valuable but exceeds the current delivery scope once governance and evaluation enforcement are in place.
- `0023` Paper Figures, Tables, and Reporting Pack
  Deferred because paper-production assets belong to the final write-up phase and are intentionally postponed until the method scope is frozen.
- `0025` Compute Budget and Fairness Policy
  Deferred because strict compute-governance formalization is useful for publication-grade benchmarking, but not required to stabilize the final thesis delivery.

## Notes

- Original task definitions are preserved under `src/cnn/ai/active_tasks/`.
- This roadmap intentionally separates implemented infrastructure from unfinished robustness methodology.
- “Deferred” does not mean discarded; it means intentionally postponed to preserve scope control and thesis stability.
