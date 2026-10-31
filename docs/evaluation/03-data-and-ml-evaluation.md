# Stetho --- Data, Training, Calibration and Evaluation Plan

## 1. Dataset scope

Initial benchmark: original MIMII 1.0, starting with fan recordings. The
original dataset contains multichannel recordings. Inspect the actual
files and metadata before fixing channel policy or split logic.

The public MIMII benchmark is not proof that a model generalizes to
every physical fan. Report benchmark results separately from recordings
collected on the real demo fan.

## 2. Evaluation tracks

### Track A --- Machine-specific

For each available fan ID:

1.  Split original normal recording files into training,
    development/calibration, and final-test partitions.
2.  Train only on normal training recordings.
3.  Use separate normal development recordings for model selection and
    threshold calibration.
4.  Evaluate on unseen normal and anomalous final-test recordings.
5.  Report each machine ID separately and an aggregate.

### Track B --- Cross-machine

Use a leave-one-machine-ID-out protocol:

1.  Train on normal recordings from the training machine IDs.
2.  Hold one machine ID out from model training.
3.  Evaluate zero-shot on the held-out machine without target-machine
    normal calibration.
4.  Separately evaluate few-shot adaptation using a predefined small set
    of target-machine normal recordings (for example, 5 or 10 clips).
5.  Never use held-out target anomalies for fitting or calibration.

State clearly that cross-ID evaluation within MIMII fan IDs is not
equivalent to generalization to an entirely new machine type,
manufacturer, site, or environment.

## 3. Leakage prevention

-   Split original files before generating overlapping windows.
-   Keep all windows from one original file in the same partition.
-   Fit feature normalization only on allowed training data.
-   Fit calibration only on designated normal calibration data.
-   Do not tune thresholds or architecture on the final test set.
-   Keep a machine-readable split manifest and dataset hash.
-   Keep official-style benchmark reproduction separate from the
    product's sliding-window evaluation.

## 4. Channel policy

Evaluate on development data:

-   A selected individual channel (including channel 0 as a reference).
-   Each channel independently with a documented aggregation policy.
-   A downmixed signal, if justified.

Do not assume that averaging eight channels is optimal. Freeze the
chosen policy before final testing.

## 5. Metrics

  ------------------------------------------------------------------------------
  Metric                              What it tells us
  ----------------------------------- ------------------------------------------
  ROC-AUC                             Ranking separation between normal and
                                      anomalous recordings

  pAUC, max FPR 0.1                   Ranking performance in the
                                      low-false-positive region

  Recall at selected FPR              Detection sensitivity at an explicit
                                      operating point

  False-positive rate                 How often normal recordings are flagged

  Precision                           Fraction of flagged recordings that are
                                      anomalous under the test labels

  CPU/GPU latency                     Inference speed

  End-to-end latency                  Upload/queue/preprocess/inference/result
                                      time

  Peak VRAM/RAM                       Resource requirements

  Model size                          Deployment footprint
  ------------------------------------------------------------------------------

Always state the evaluation split, sample counts, machine IDs, and
threshold-selection method. A high ROC-AUC does not by itself establish
a low operational false-positive rate.

## 6. Calibration and honest score presentation

Primary model output: reconstruction error.

Calibrate the score using a separate set of normal recordings from the
relevant machine. Choose an operating threshold on
development/calibration data according to a documented false-positive
target or other justified operating requirement.

A normal-reference percentile is not a probability of failure. Avoid
presenting an uncalibrated neural-network output as confidence.

Use three result states:

-   **Normal-like:** score is within the calibrated normal range.
-   **Unusual:** score exceeds the chosen threshold.
-   **Inconclusive:** invalid/poor-quality audio, insufficient
    calibration, or unsupported conditions.

For very small enrollment sets, label the baseline provisional and avoid
implying statistically strong guarantees. Five clips is a minimum
provisional enrollment target; ten or more comparable normal clips are
preferable, but more may be needed to represent real operating
variability.

## 7. Required experiment comparison

  -----------------------------------------------------------------------
  Experiment                          Purpose
  ----------------------------------- -----------------------------------
  Simple baseline                     Reference point
  detector/autoencoder                

  Stetho-MSAE                         Main proposed model

  Stetho-MSAE without dilation        Test whether multi-scale temporal
                                      context helps

  Latent size 64 vs 128               Test capacity vs speed/resource
                                      trade-off

  L1 vs MSE                           Test reconstruction objective
  -----------------------------------------------------------------------

Do not add more model families unless the results reveal a specific
limitation that the added model can reasonably address.

## 8. Performance measurement

Measure separately:

-   Model-only inference latency.
-   Preprocessing plus model latency.
-   End-to-end application latency.
-   CPU latency.
-   GPU latency on the RTX 3050 4 GB.
-   Peak VRAM and RAM.
-   Throughput for the selected window/overlap policy.

Report hardware, software versions, batch size, audio duration, warm-up
policy, and number of repetitions. Do not claim a real-time latency
target until it is measured on the actual deployment environment.
