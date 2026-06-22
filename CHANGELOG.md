# Changelog

All notable changes to NRT-Defense will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-22

### Added
- Per-Turn Analyzer — Message analysis: keywords, patterns, channel risk, turn escalation
- Adaptive Misdirection Engine — Defense action selection: PASS/MONITOR/SANITIZE/MISDIRECT/BLOCK/ALERT
- CSF State Monitor — Real-time tracking of 6 Critical Safety Functions
- Multi-Model Vulnerability Mapper — Disjoint vulnerability tracking + model robustness ranking
- CLI — `nrt-audit` command
- 57 tests (unit + integration)
- CI/CD via GitHub Actions (Python 3.10, 3.11, 3.12)
- Makefile with standard targets
- ruff, mypy, black configuration
- Coverage configuration (minimum 80%)
- SECURITY.md and CHANGELOG.md

### Results (from NRT-Bench paper)

| Metric | Value |
|--------|-------|
| Attack success rate | 8.7% — 12.1% |
| Sessions analyzed | 149 |
| Models evaluated | 4 frontier LLMs |
| Vulnerability overlap | Nearly disjoint |

**NRT-Defense target:** Reduce attack success to <1%.
