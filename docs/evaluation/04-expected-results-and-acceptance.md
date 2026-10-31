# Stetho --- Expected Results and Acceptance Criteria

**Important:** The values in this document are project targets and
validation expectations, not achieved results. Do not put them on a
public demo, resume, or hackathon submission as measured achievements
until experiments confirm them.

## 1. Expected product outcome

A judge should be able to:

1.  Open the deployed Stetho application.
2.  Create a profile for a physical machine.
3.  Upload or record normal audio to enroll the machine.
4.  See enrollment validation and baseline status.
5.  Submit a new recording.
6.  Receive a Normal-like, Unusual, or Inconclusive result.
7.  Inspect the anomaly score and calibrated threshold/reference.
8.  View waveform and spectrogram evidence.
9.  Review a report with recording metadata, model version, and
    limitations.

The product identifies acoustic deviations from a machine-specific
normal baseline. It does not diagnose a specific mechanical fault or
predict failure.

## 2. ML expected outcomes

  -----------------------------------------------------------------------
  Area                    Expected outcome        How to verify
  ----------------------- ----------------------- -----------------------
  Training                Reproducible            Repeat with recorded
                          normal-only training    config/seed

  Detection               Stetho-MSAE is          Held-out file-level
                          competitive with the    benchmark
                          baseline                

  Machine-specific use    Separate results per    Per-ID metrics
                          fan ID                  

  Cross-machine use       Quantified zero-shot    Leave-one-ID-out
                          and few-shot            evaluation
                          performance             

  Calibration             Threshold derived       Inspect manifests and
                          without final-test      calibration code
                          leakage                 

  Reliability             Invalid audio becomes   Negative tests
                          Inconclusive, not a     
                          misleading prediction   

  Interpretability        Show score,             UI acceptance test
                          threshold/reference,    
                          waveform and            
                          spectrogram             
  -----------------------------------------------------------------------

## 3. Proposed performance targets

These are internal goals, not promises or guaranteed outcomes.

  -----------------------------------------------------------------------
  Measure                             Initial engineering target
  ----------------------------------- -----------------------------------
  ROC-AUC                             Aim for at least 0.80 on a clearly
                                      defined held-out benchmark

  pAUC at max FPR 0.1                 Improve over the baseline; report
                                      actual value

  False-positive rate                 Target 5% on a representative
                                      validation set, then report
                                      measured test FPR

  CPU inference                       Complete a 1-second window
                                      comfortably within the product's
                                      update interval on the deployment
                                      CPU

  GPU inference                       Measure on RTX 3050 4 GB; avoid
                                      assuming a number before
                                      benchmarking

  Model footprint                     Keep compact enough for CPU
                                      deployment; record exact artifact
                                      size

  Reliability                         No silent failures; invalid input
                                      and model errors produce explicit
                                      states
  -----------------------------------------------------------------------

A 5% calibration target does not guarantee 5% production false
positives. Real operating conditions can differ from the calibration
sample.

## 4. Deployment acceptance

-   [ ] Deployed frontend loads over HTTPS.
-   [ ] Backend readiness confirms model and database availability.
-   [ ] Machine profiles persist.
-   [ ] Audio uploads persist privately.
-   [ ] Enrollment and analysis work after a fresh deployment.
-   [ ] CPU inference works on the actual host.
-   [ ] Analysis status is recoverable after page refresh.
-   [ ] Errors are visible and actionable.
-   [ ] The model and calibration versions are visible in result
    metadata.
-   [ ] The demo can be completed from a clean browser session.

## 5. Hackathon demo acceptance

Prepare two paths:

### Live path

Use the real demo fan, collect/enroll normal audio, then record a
changed acoustic condition and analyze it. Explain that the result
indicates deviation, not a confirmed fault.

### Fallback path

Use clearly labelled prerecorded audio and stored example results if the
live device, microphone, network, or hosted inference service fails.
Never present prerecorded output as a live prediction.

## 6. Final reporting template

Fill in only after running experiments:

  Result                                 Measured value
  -------------------------------------- ----------------
  Dataset version and manifest hash      TBD
  Machine IDs                            TBD
  Training/validation/test file counts   TBD
  ROC-AUC per machine                    TBD
  pAUC per machine                       TBD
  Recall at selected FPR                 TBD
  Measured test FPR                      TBD
  CPU model latency                      TBD
  GPU model latency                      TBD
  End-to-end deployed latency            TBD
  Peak VRAM/RAM                          TBD
  Model artifact size                    TBD
  Deployment URL                         TBD

## 7. Claims we must not make without evidence

-   "99% accurate" without a clearly defined, valid evaluation.
-   "Predicts machine failure."
-   "Diagnoses bearing, motor, or blade faults."
-   "Works on every machine."
-   "Real-time" without measured end-to-end latency under stated
    conditions.
-   "98.5% confidence" when the number is only a normal-reference
    percentile.
-   "Cross-domain generalization" based only on holding out one MIMII
    fan ID.
