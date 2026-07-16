# Epic 6: Differential Privacy Implementation — Requirements

## Overview

This epic implements the differential privacy layer for the Kizuna Multimodal Privacy Engine. The privacy layer ensures APPI compliance by adding calibrated noise to embedding vectors, tracking privacy budgets, and providing mathematical guarantees that individual sensor readings cannot be reverse-engineered from the published embeddings.

## Business Context

### Problem Statement

The Kizuna engine processes sensitive multimodal data (video, audio, environmental sensors) in privacy-critical environments such as:
- Nursing facilities monitoring elderly residents
- Railway stations tracking crowd flow
- Smart buildings collecting behavioral patterns

While the system already converts raw data to embeddings (preventing direct reconstruction), additional protection is needed to:
1. **Prevent statistical inference attacks** — Even if raw data cannot be recovered, analyzing patterns across many embeddings could reveal individual behaviors
2. **Ensure APPI compliance** — Japan's Act on the Protection of Personal Information requires demonstrable privacy guarantees for biometric and behavioral data
3. **Enable data sharing** — Organizations need mathematical proof that shared embeddings preserve privacy, allowing cross-domain collaboration
4. **Provide configurable privacy levels** — Different deployment scenarios require different privacy-utility tradeoffs

### Target Users

- **System Administrators**: Configure privacy parameters (ε, δ) based on regulatory requirements
- **Data Scientists**: Analyze embeddings for anomaly detection while respecting privacy budgets
- **Compliance Officers**: Audit privacy guarantees and budget consumption
- **Edge Node Operators**: Deploy privacy-preserving embeddings at resource-constrained edge devices

### Success Criteria

1. **Functional Completeness**: All embeddings can be protected with differential privacy before storage or transmission
2. **APPI Compliance**: System provides formal (ε, δ)-differential privacy guarantees with configurable parameters
3. **Privacy-Utility Balance**: Anomaly detection accuracy remains ≥80% with ε ≤ 1.0
4. **Performance**: Privacy layer adds <10ms latency per embedding on edge devices (2 CPU cores)
5. **Budget Management**: System prevents privacy budget exhaustion through tracking and alerting
6. **Auditability**: All privacy operations are logged with timestamps, parameters, and budget impact

## Functional Requirements

### FR-1: Laplace Mechanism for Embedding Protection

**Description**: Implement ε-differential privacy using the Laplace mechanism to add calibrated noise to embedding vectors.

**Rationale**: The Laplace mechanism is the standard approach for (ε)-DP. It provides pure differential privacy without requiring a δ parameter, simplifying compliance verification.

**Requirements**:
- **FR-1.1**: Accept embedding vector (numpy array), epsilon (ε), and sensitivity (Δ) as inputs
- **FR-1.2**: Compute Laplace scale parameter: b = Δ / ε
- **FR-1.3**: Generate i.i.d. Laplace noise for each vector dimension: noise ~ Laplace(0, b)
- **FR-1.4**: Add noise element-wise to embedding: noisy_vector = original_vector + noise
- **FR-1.5**: Return noisy vector with same shape and dtype as input
- **FR-1.6**: Validate inputs: ε > 0, Δ > 0, vector is finite (no NaN, Inf)
- **FR-1.7**: Raise clear exceptions for invalid inputs with actionable error messages

**Acceptance Criteria**:
- Given a 512-dimensional embedding with ε=1.0 and Δ=2.0, noise is added correctly
- Output vector has same shape (512,) as input
- Statistical tests confirm noise follows Laplace distribution with scale b = 2.0
- Invalid inputs (ε≤0, NaN vector) raise ValueError with descriptive message

---

### FR-2: Gaussian Mechanism for Embedding Protection

**Description**: Implement (ε, δ)-differential privacy using the Gaussian mechanism as an alternative to Laplace.

**Rationale**: The Gaussian mechanism is preferred in some scenarios:
- When composition theorems (e.g., zCDP, Rényi DP) are used for tighter budget accounting
- When working with other systems that use Gaussian DP
- When the utility-privacy tradeoff is better for specific workloads

**Requirements**:
- **FR-2.1**: Accept embedding vector, epsilon (ε), delta (δ), and sensitivity (Δ) as inputs
- **FR-2.2**: Compute Gaussian standard deviation: σ = Δ · sqrt(2 · ln(1.25 / δ)) / ε
- **FR-2.3**: Generate i.i.d. Gaussian noise: noise ~ N(0, σ²)
- **FR-2.4**: Add noise element-wise to embedding
- **FR-2.5**: Return noisy vector with same shape and dtype
- **FR-2.6**: Validate inputs: ε > 0, 0 < δ < 1, Δ > 0, vector is finite
- **FR-2.7**: Support configuration flag to choose between Laplace and Gaussian via config

**Acceptance Criteria**:
- Given ε=1.0, δ=1e-5, Δ=2.0, correct σ is computed
- Noise follows Gaussian distribution (verified via statistical tests)
- Config flag `privacy.mechanism` supports values "laplace" and "gaussian"
- System defaults to Laplace if not specified

---

### FR-3: Privacy Budget Tracking

**Description**: Track cumulative privacy budget consumption across all embedding operations to prevent privacy budget exhaustion.

**Rationale**: Differential privacy guarantees degrade under composition. If many queries use the same data, the total privacy loss is the sum of individual losses (sequential composition). Without tracking, the system could exceed acceptable privacy thresholds.

**Requirements**:
- **FR-3.1**: Maintain a `PrivacyBudgetTracker` that records ε spent per operation
- **FR-3.2**: Implement sequential composition: total_ε = Σ εᵢ across all operations
- **FR-3.3**: Support configurable total budget ceiling (default: ε_total = 10.0)
- **FR-3.4**: Before each operation, check if adding εᵢ would exceed budget
- **FR-3.5**: If budget exceeded, raise `PrivacyBudgetExhaustedError` and refuse operation
- **FR-3.6**: Issue warning log when budget reaches 80% of ceiling
- **FR-3.7**: Persist budget state to JSON file for recovery after process restarts
- **FR-3.8**: Support budget reset (with logged audit trail) for new data collection periods
- **FR-3.9**: Implement parallel composition: if k independent datasets are processed simultaneously, total_ε = max(εᵢ) not Σ εᵢ
- **FR-3.10**: Provide query methods: `get_remaining_budget()`, `get_budget_history()`

**Acceptance Criteria**:
- System starts with ε_remaining = 10.0
- After 5 operations with ε=1.0 each, ε_remaining = 5.0
- After 10 operations with ε=1.0 each, next operation raises `PrivacyBudgetExhaustedError`
- Warning logged at operation 8 (80% threshold: 8.0 of 10.0 consumed)
- After process restart, budget state is recovered from JSON file
- Parallel composition correctly computes max(εᵢ) for independent data sources

---

### FR-4: Sensitivity Calibration

**Description**: Empirically measure the L2 sensitivity of the embedding function to determine appropriate Δ values for noise calibration.

**Rationale**: The sensitivity Δ = max ‖f(x) - f(x')‖₂ over all adjacent inputs determines how much noise is needed. Overestimating Δ adds excessive noise (poor utility); underestimating Δ violates privacy guarantees. Empirical calibration balances both.

**Requirements**:
- **FR-4.1**: Create `SensitivityCalibrator` class that measures empirical sensitivity
- **FR-4.2**: Generate 10,000+ pairs of adjacent sensor payloads (payloads differing by one sensor reading)
- **FR-4.3**: For each pair (x, x'), compute embeddings: e = f(x), e' = f(x')
- **FR-4.4**: Compute L2 distance: d = ‖e - e'‖₂
- **FR-4.5**: Estimate sensitivity: Δ_empirical = max(d) across all pairs
- **FR-4.6**: Add safety margin: Δ_calibrated = 1.1 × Δ_empirical (10% buffer)
- **FR-4.7**: Log calibration results: min/max/mean/median distances, Δ_calibrated
- **FR-4.8**: Store Δ_calibrated in config file for use by DP mechanisms
- **FR-4.9**: Provide confidence intervals using bootstrap resampling
- **FR-4.10**: Support re-calibration when embedding models are updated

**Acceptance Criteria**:
- Calibration script runs on 10,000+ synthetic payload pairs
- Δ_calibrated is stored in `config/default.yaml` under `privacy.sensitivity`
- Calibration report logged with statistics (min, max, mean, P95, P99 distances)
- Confidence interval computed: e.g., "Δ_calibrated = 2.34 ± 0.12 (95% CI)"
- Subsequent privacy operations use Δ_calibrated from config

---

### FR-5: Property-Based Testing for DP Mechanisms

**Description**: Implement property-based tests using Hypothesis to verify statistical properties of the differential privacy mechanisms.

**Rationale**: Unit tests with fixed inputs cannot verify that noise distributions match theoretical expectations. Property-based tests generate thousands of random inputs and verify invariants hold across all cases.

**Requirements**:
- **FR-5.1**: Test Laplace noise mean preservation
  - Generate 10,000+ noisy samples for random vectors
  - Verify: mean(noisy_samples) ≈ original_vector (within 3σ statistical tolerance)
- **FR-5.2**: Test Laplace noise variance scaling
  - Verify: var(noise) ≈ 2b² where b = Δ/ε
  - Test across multiple ε values: [0.1, 1.0, 10.0]
  - Verify: higher ε → lower variance (less noise)
- **FR-5.3**: Test Laplace distribution shape
  - Use Kolmogorov-Smirnov test to verify noise follows Laplace(0, b)
  - Significance level: α = 0.05
- **FR-5.4**: Test Gaussian noise properties
  - Verify mean preservation and variance: σ² = (Δ · sqrt(2 ln(1.25/δ)) / ε)²
  - Use KS test for Gaussian distribution
- **FR-5.5**: Test sensitivity-noise relationship
  - Given fixed ε, doubling Δ should double noise scale
- **FR-5.6**: Test privacy parameter validation
  - Property: all invalid combinations (ε≤0, δ≥1, etc.) raise exceptions

**Acceptance Criteria**:
- Property tests run with Hypothesis generating ≥1,000 test cases per property
- All statistical tests pass with p-value > 0.05
- Tests cover edge cases: very small ε (0.01), very large vectors (10,000-dim), zero vectors
- Test suite completes in <60 seconds

---

### FR-6: Unit Testing for Privacy Budget Tracker

**Description**: Comprehensive unit tests for the privacy budget tracking system.

**Requirements**:
- **FR-6.1**: Test sequential composition accumulation
  - Multiple operations correctly sum ε values
- **FR-6.2**: Test budget ceiling enforcement
  - Operation refused when budget exceeded
  - Correct exception type and message
- **FR-6.3**: Test budget persistence
  - State saved to JSON after each operation
  - State correctly restored after simulated restart
- **FR-6.4**: Test 80% threshold warning
  - Warning logged at exactly 80% consumption
  - No warning before 80%, warning present after
- **FR-6.5**: Test parallel composition
  - Independent data sources use max(εᵢ) not sum
- **FR-6.6**: Test budget queries
  - `get_remaining_budget()` returns correct value
  - `get_budget_history()` returns all operations

**Acceptance Criteria**:
- 100% branch coverage for `PrivacyBudgetTracker` class
- All tests pass in <5 seconds
- Tests verify both happy path and error conditions

---

## Non-Functional Requirements

### NFR-1: Performance

- **NFR-1.1**: Noise addition adds <10ms latency per embedding on edge constraints (2 CPU cores, 2GB RAM)
- **NFR-1.2**: Budget tracking adds <1ms overhead per operation
- **NFR-1.3**: Sensitivity calibration completes in <5 minutes for 10,000 samples
- **NFR-1.4**: All DP operations are thread-safe for multi-threaded embedding pipeline

**Measurement**: Benchmark with `time.perf_counter_ns()` on target hardware constraints

### NFR-2: Memory Efficiency

- **NFR-2.1**: Noise generation does not allocate temporary arrays >2× embedding size
- **NFR-2.2**: Budget tracker memory footprint <1MB for 100,000 operations
- **NFR-2.3**: Calibration memory <500MB peak during 10,000 sample run

**Measurement**: Profile with `tracemalloc` and memory_profiler

### NFR-3: Accuracy Preservation

- **NFR-3.1**: With ε=1.0, anomaly detection accuracy ≥80% (baseline: 85% without DP)
- **NFR-3.2**: With ε=0.1, anomaly detection accuracy ≥70%
- **NFR-3.3**: Cosine similarity between noisy and original embeddings: ≥0.90 for ε=1.0
- **NFR-3.4**: L2 distance increase due to noise: <20% for ε=1.0

**Measurement**: Compare anomaly detection metrics (recall, precision, F1) with/without DP on labeled test set

### NFR-4: Configurability

- **NFR-4.1**: All privacy parameters configurable via `config/default.yaml`
- **NFR-4.2**: Support runtime override via environment variables: `KIZUNA_PRIVACY_EPSILON`, `KIZUNA_PRIVACY_DELTA`, `KIZUNA_PRIVACY_MECHANISM`
- **NFR-4.3**: Support per-modality sensitivity values (video, audio, sensor may have different Δ)
- **NFR-4.4**: Support disabling DP for development/testing (with prominent warning logs)

### NFR-5: Auditability

- **NFR-5.1**: All DP operations logged with structured fields: `{"event": "dp_noise_added", "epsilon": 1.0, "sensitivity": 2.0, "mechanism": "laplace", "timestamp": "...", "embedding_id": "..."}`
- **NFR-5.2**: Budget operations logged: `{"event": "budget_consumed", "epsilon_spent": 1.0, "epsilon_remaining": 9.0, ...}`
- **NFR-5.3**: Calibration results persisted to `logs/privacy_calibration.json`
- **NFR-5.4**: All logs include correlation IDs for tracing operations end-to-end

### NFR-6: Error Handling

- **NFR-6.1**: All exceptions inherit from base `KizunaPrivacyError` class
- **NFR-6.2**: Exception messages include actionable guidance (e.g., "Budget exhausted. Consider increasing privacy.budget_ceiling in config or resetting budget for new collection period.")
- **NFR-6.3**: Invalid inputs never cause crashes — always raise exceptions with context
- **NFR-6.4**: Failed budget file I/O triggers warning but does not crash (fallback to in-memory tracking)

### NFR-7: APPI Compliance Documentation

- **NFR-7.1**: Document formal privacy guarantee: "Embeddings satisfy (ε, δ)-differential privacy with ε≤1.0, δ≤1e-5"
- **NFR-7.2**: Provide privacy loss accounting report generation for auditors
- **NFR-7.3**: Document composition theorems used (sequential, parallel)
- **NFR-7.4**: Include references to academic papers for Laplace/Gaussian mechanisms

---

## Technical Constraints

### TC-1: Dependencies
- Python 3.10+
- NumPy for array operations
- SciPy for statistical distributions (Laplace, Gaussian)
- Hypothesis for property-based testing
- Pytest for unit testing

### TC-2: Integration Points
- Must integrate with `EmbeddingEngine` from Epic 5
- Must integrate with `VectorStore` for noisy embedding storage (Epic 8)
- Must integrate with `ConfigManager` from Epic 1

### TC-3: Edge Device Constraints
- All operations must work with INT8 embeddings (not just FP32)
- No GPU required — CPU-only implementation
- No network calls during noise addition (must be offline-capable)

---

## User Stories

### US-1: System Administrator Configures Privacy Parameters
**As a** system administrator  
**I want to** configure epsilon and delta values via config file  
**So that** I can balance privacy protection and anomaly detection accuracy for my deployment

**Acceptance Criteria**:
- Edit `config/default.yaml` to set `privacy.epsilon: 1.0` and `privacy.delta: 1e-5`
- Restart system or reload config
- All subsequent embeddings use new parameters
- Logs confirm: "Loaded privacy config: epsilon=1.0, delta=1e-5, mechanism=laplace"

---

### US-2: Data Scientist Monitors Privacy Budget
**As a** data scientist  
**I want to** query remaining privacy budget before running anomaly detection  
**So that** I can avoid exceeding privacy limits during experiments

**Acceptance Criteria**:
- Call `tracker.get_remaining_budget()` → returns float (e.g., 6.5)
- Call `tracker.get_budget_history()` → returns list of operations with ε spent
- Budget displayed in dashboard with progress bar (e.g., "Budget: 3.5 / 10.0 (35% remaining)")

---

### US-3: Compliance Officer Audits Privacy Guarantees
**As a** compliance officer  
**I want to** generate a privacy accounting report  
**So that** I can demonstrate APPI compliance to regulators

**Acceptance Criteria**:
- Run command: `python scripts/generate_privacy_report.py --output audit.json`
- Report includes:
  - Total privacy budget: ε_total = 10.0
  - Budget consumed: ε_consumed = 7.2
  - Number of operations: 72 (each ε=0.1)
  - Composition method: Sequential
  - Formal guarantee: "(7.2, 1e-5)-differential privacy"
  - Timestamp and git commit hash

---

### US-4: Developer Debugs Privacy Layer
**As a** developer  
**I want to** disable differential privacy temporarily  
**So that** I can debug anomaly detection without noise interference

**Acceptance Criteria**:
- Set config: `privacy.enabled: false`
- System logs warning: "⚠️ PRIVACY DISABLED — Do not use in production!"
- Embeddings passed through without noise
- Budget tracking still active (logs warning for each operation)

---

### US-5: Operator Re-calibrates Sensitivity After Model Update
**As an** operations engineer  
**I want to** re-run sensitivity calibration after updating embedding models  
**So that** noise levels remain appropriate for the new model

**Acceptance Criteria**:
- Run: `python scripts/calibrate_sensitivity.py`
- Calibration uses 10,000+ synthetic payloads
- New Δ_calibrated written to config file
- Calibration report saved to `logs/privacy_calibration_2026-07-14.json`
- Subsequent embeddings use new Δ_calibrated

---

## Assumptions

1. **Embedding vectors are L2-normalized**: Sensitivity analysis assumes unit-norm vectors (‖v‖₂ = 1)
2. **Edge nodes have synchronized clocks**: Budget tracking across distributed nodes requires accurate timestamps
3. **Config file is read-only during runtime**: Changes require restart or explicit reload
4. **NumPy random seed can be controlled**: For reproducible testing
5. **Budget state file is on persistent storage**: Not on tmpfs or RAM disk

---

## Risks and Mitigations

### Risk 1: Excessive Noise Degrades Anomaly Detection Accuracy
**Impact**: High — If accuracy drops below 70%, the system becomes unusable  
**Likelihood**: Medium — Depends on calibrated sensitivity Δ  
**Mitigation**:
- Run accuracy benchmarks with various ε values during development
- Provide clear documentation: "For ε < 0.5, expect accuracy degradation"
- Implement adaptive noise scaling based on anomaly type criticality

### Risk 2: Privacy Budget Exhaustion in Production
**Impact**: High — System stops processing embeddings  
**Likelihood**: Low — With monitoring and alerting  
**Mitigation**:
- Dashboard displays real-time budget consumption
- Alert at 80% threshold (configurable)
- Document budget reset procedure for administrators
- Consider advanced composition (zCDP, Rényi DP) for tighter accounting

### Risk 3: Sensitivity Underestimation Violates Privacy Guarantees
**Impact**: Critical — Privacy guarantees invalid  
**Likelihood**: Low — Calibration includes 10% safety margin  
**Mitigation**:
- Use conservative estimates (max + margin)
- Re-calibrate when models change
- Formal sensitivity analysis for critical deployments
- Document assumptions in calibration report

### Risk 4: Performance Overhead Exceeds 10ms Target
**Impact**: Medium — May require optimization  
**Likelihood**: Low — Noise addition is O(D) where D=512  
**Mitigation**:
- Use NumPy vectorized operations (no Python loops)
- Pre-allocate noise buffers
- Benchmark on target hardware early
- Consider C++ implementation if needed

---

## Open Questions

1. **Q**: Should we implement advanced composition (zCDP, Rényi DP) for tighter budget accounting?  
   **A**: Not in Epic 6 — start with basic sequential composition. Advanced composition can be Epic 11 enhancement.

2. **Q**: How should we handle budget across multiple edge nodes (distributed setting)?  
   **A**: Each edge node maintains independent budget. Central aggregator can optionally enforce global budget (future work).

3. **Q**: Should budget reset be automatic (e.g., every 30 days) or manual?  
   **A**: Manual with audit logging for Epic 6. Automatic reset with CRON can be added later.

4. **Q**: Do we need per-user privacy budgets (if system tracks individuals)?  
   **A**: No — Kizuna is designed for aggregate monitoring, not individual tracking. Per-dataset budget is sufficient.

5. **Q**: Should we support other DP mechanisms (e.g., exponential mechanism for categorical outputs)?  
   **A**: Not needed for Epic 6 — embeddings are continuous vectors. Laplace + Gaussian sufficient.

---

## Success Metrics

### Implementation Completeness
- [ ] All 6 functional requirements implemented
- [ ] All NFRs measured and documented
- [ ] 100% passing unit tests with ≥80% code coverage
- [ ] Property-based tests generate ≥1,000 test cases per property

### Privacy Guarantees
- [ ] Formal (ε, δ)-DP guarantee documented with proof sketch
- [ ] Sensitivity calibration report generated
- [ ] Budget tracking prevents overrun in stress tests

### Performance
- [ ] Noise addition: <10ms per embedding (P95)
- [ ] Budget tracking: <1ms per operation (P95)
- [ ] Memory: <500MB during calibration

### Utility
- [ ] Anomaly detection accuracy ≥80% with ε=1.0
- [ ] Cosine similarity ≥0.90 between original and noisy embeddings (ε=1.0)

### Documentation
- [ ] APPI compliance report generated
- [ ] Sensitivity calibration guide written
- [ ] User guide for configuration and monitoring
- [ ] Privacy accounting examples documented

---

## Out of Scope for Epic 6

The following items are explicitly **not** part of Epic 6 and will be addressed in later epics:

1. **Raw Data Destruction** — Memory wiping is Epic 7
2. **Vector Database Integration** — Qdrant/FAISS integration is Epic 8
3. **Anomaly Detection** — Detection algorithms are Epic 9
4. **Edge Simulation** — Multi-node Docker setup is Epic 11
5. **Dashboard** — Visualization is Epic 13
6. **Advanced Composition** — zCDP/Rényi DP is future work
7. **Distributed Budget Tracking** — Cross-node budget coordination is future work
8. **Adaptive Privacy** — Dynamic ε adjustment based on threat models is future work

---

## Dependencies

### Epic 5 (Completed)
- `EmbeddingEngine` produces `UnifiedEmbedding` objects
- Embedding vectors have shape `(D,)` where D is configurable (default 512)
- Embeddings are L2-normalized

### Epic 1 (Completed)
- `ConfigManager` loads `config/default.yaml`
- Structured logging infrastructure available

### Future Epics
- Epic 7 will integrate DP noise before memory wiping
- Epic 8 will store noisy embeddings in Qdrant
- Epic 9 will evaluate anomaly detection accuracy with DP noise

---

## Glossary

- **DP**: Differential Privacy
- **ε (epsilon)**: Privacy budget parameter — smaller ε = stronger privacy, more noise
- **δ (delta)**: Privacy budget failure probability (for Gaussian mechanism)
- **Δ (Delta)**: Sensitivity — maximum L2 distance between embeddings of adjacent inputs
- **APPI**: Japan's Act on the Protection of Personal Information
- **Sequential Composition**: Privacy loss accumulates additively across queries
- **Parallel Composition**: Privacy loss is maximum (not sum) across independent datasets
- **L2 Norm**: Euclidean distance — ‖v‖₂ = sqrt(Σ vᵢ²)
- **KS Test**: Kolmogorov-Smirnov test for distribution matching

---

**Document Status**: Draft for Review  
**Author**: Kizuna Development Team  
**Date**: 2026-07-14  
**Version**: 1.0
