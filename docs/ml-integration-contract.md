# STETHO — ML INTEGRATION CONTRACT & MODEL BOUNDARY SPECIFICATION

**Document Version:** 1.0.0  
**Status:** ACTIVE  
**Last Updated:** October 31, 2026  

---

## 1. Architectural Boundary Principle

The frontend, UI components, and API routing layer MUST remain strictly decoupled from the internal architecture of individual machine learning models. 

Whether an anomaly score is computed by:
- `stetho_msae` (Multi-Scale Depthwise-Separable CNN Autoencoder)
- `dense_autoencoder` (Bottleneck MLP Autoencoder)
- `isolation_forest` (Tree-based ensemble)
- `memory_bank_knn` (Bottleneck latent k-NN)

The backend MUST expose a uniform inference adapter contract adhering to this specification.

---

## 2. Standardized Analysis Result Schema

Every analysis output returned to the frontend or persisted in the database MUST adhere to the following payload:

```json
{
  "analysis_id": "ana_01j7x9p4d...",
  "machine_id": "id_00",
  "created_at": "2026-10-31T18:00:00Z",
  "model": {
    "model_id": "stetho_msae",
    "version": "1.1.0",
    "checkpoint_hash": "sha256:4a8c9e...",
    "preprocessing_version": "1.0.0"
  },
  "calibration": {
    "calibration_version": "1.1.0",
    "enrolled_count": 30,
    "is_provisional": false,
    "theta_lower": 0.0035,
    "theta_upper": 0.00512
  },
  "quality": {
    "quality_status": "ACCEPTED",
    "rms_dbfs": -22.4,
    "peak_amplitude": 0.78,
    "snr_db": 28.5,
    "is_clipped": false,
    "duration_sec": 10.0,
    "sample_rate": 16000
  },
  "result": {
    "anomaly_score": 0.00341,
    "decision": "NORMAL_LIKE",
    "confidence": 0.94,
    "processing_duration_ms": 58.6
  },
  "spectral_fingerprint": {
    "spectral_centroid_hz": 1420.5,
    "band_energies": {
      "sub_bass": -42.1,
      "low": -28.3,
      "mid": -24.5,
      "high": -36.8
    }
  }
}
```

---

## 3. Decision Semantics Contract

The decision engine guarantees strictly three categorical outputs:

1. **`NORMAL_LIKE`**:
   - Condition: `anomaly_score <= theta_upper` AND `quality_status != "REJECTED"`.
   - Meaning: Machine acoustic characteristics align with enrolled normal profile. No anomalous acoustic energy detected.
2. **`UNUSUAL`**:
   - Condition: `anomaly_score > theta_upper` AND `quality_status != "REJECTED"`.
   - Meaning: Machine acoustic signature exhibits unusual deviation beyond normal calibration threshold. Recommended action: physical inspection.
3. **`INCONCLUSIVE`**:
   - Condition: High recording noise, quality warning, or score in the borderline zone with provisional calibration ($N < 30$).
   - Meaning: Reliable classification is not possible. Operator advised to re-record in quiet conditions.

---

## 4. Version Compatibility Rule

- Models trained with Preprocessing Version $X$ CANNOT be evaluated with Preprocessing Version $Y$ without explicit adapter transformation.
- Incompatible calibration profiles MUST be flagged with `is_provisional = true` and a calibration mismatch warning.
