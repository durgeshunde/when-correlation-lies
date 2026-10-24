# Pre-Sprint Work Log

## Purpose
This document records what was created, modified, and decided BEFORE the 
official sprint start on October 23, 2026. Nothing in this log should be 
presented as work done during the sprint.

## Timeline

### Before Sprint Start (Pre-October 23, 2026)
- **Nothing.** This project directory did not exist before the sprint.
- No code, no prior experimental results, no partial implementations.
- The project is created during the sprint.

### Sprint Start: October 23, 2026, ~20:15 IST (14:45 UTC)
The sprint officially began at this time. All code, experiments, and analysis 
in this repository were created after this timestamp.

## Pre-Existing Work (from Other Sources)

### Primary method (NOT our invention):
- Bonjour, T., Aggarwal, V. & Bhargava, B. (2022). "Information-theoretic 
  approach to detect collusion in multi-agent games." PMLR 180:223–232.
- This paper defined pairwise net influence, the experimental game setup 
  (RPS), and the collusion detection framework.
- Our implementation faithfully follows their method.

### iterated RPS game design:
- Standard game; game theory literature.
- No specific prior code reused.

### llm-agent-collusion repository:
- https://github.com/andyjliu/llm-agent-collusion
- Inspected for reference (repository structure, supported models).
- NOT incorporated into our code — we built an independent implementation.
- Cited as related work.

## What Is New in This Project
- Independent Python implementation of the game environment and agents.
- Independent implementation of pairwise net influence and baseline detectors.
- Controlled experimental conditions (A, B, C, D) with documented design rationale.
- Evaluation pipeline with bootstrap CIs.
- Analysis scripts and figures.
- Research report.

---

*This log is maintained for transparency and academic integrity.*
