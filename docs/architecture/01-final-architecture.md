# Stetho --- Final Architecture

**Status:** Proposed architecture for implementation\
**Product:** Machine-specific acoustic anomaly monitoring and inspection
assistant\
**Primary demo machine:** Small electric fan\
**Benchmark dataset:** Original MIMII 1.0\
**Hardware target:** NVIDIA RTX 3050 Laptop GPU, 4 GB VRAM; CPU fallback
required

## 1. Architecture decision

Stetho will use a **shared multi-scale convolutional autoencoder**,
referred to as **Stetho-MSAE**.

The design preserves multi-scale acoustic learning from the original
three-branch proposal but avoids three independent feature extractors.
Multi-scale temporal context is learned inside one shared encoder using
compact residual depthwise-separable convolutions and dilated
convolutions.

The initial production detector has one primary anomaly signal:
reconstruction error. Machine-specific calibration converts raw errors
into an interpretable anomaly score and decision. Latent-distance
scoring, attention fusion, VAE, Transformer, BEATs, PANNs, and ensembles
are excluded from the initial implementation unless measured evidence
justifies adding them.

## 2. System architecture

``` mermaid
flowchart TD
    U[User / Judge] --> WEB[Next.js Web Application]
    WEB --> API[FastAPI Application API]
    API --> AUTH[Authentication and Request Validation]
    API --> DB[(PostgreSQL: profiles, jobs, results)]
    API --> OBJ[(Private Object Storage: audio and reports)]
    API --> JOB[Analysis Job]
    JOB --> PRE[Audio Validation and Preprocessing]
    PRE --> FE[Single Log-Mel Frontend]
    FE --> ENC[Shared Multi-Scale CNN Encoder]
    ENC --> LAT[Compact Latent Representation]
    LAT --> DEC[Lightweight CNN Decoder]
    DEC --> ERR[Reconstruction Error]
    ERR --> CAL[Machine-Specific Calibration]
    CAL --> RES[Normal-like / Unusual / Inconclusive]
    RES --> DB
    RES --> WEB
```

## 3. ML model architecture

``` mermaid
flowchart TD
    A[Audio: 16 kHz] --> B[Channel policy and quality checks]
    B --> C[Mono float32 model input]
    C --> D[Single log-Mel spectrogram]
    D --> E[Compact convolutional stem]
    E --> F[Depthwise-separable residual block]
    F --> G[Residual dilated convolution blocks]
    G --> H[Latent embedding: start at 64 dimensions]
    H --> I[Lightweight CNN decoder]
    I --> J[Reconstructed log-Mel]
    D --> K[Reconstruction error]
    J --> K
    K --> L[Machine-specific normal calibration]
    L --> M[Anomaly score and decision]
```

## 4. Model components and responsibilities

  -----------------------------------------------------------------------
  Component               Responsibility          Design choice
  ----------------------- ----------------------- -----------------------
  Audio validation        Reject corrupt, silent, Lightweight
                          clipped, too-short, or  deterministic checks
                          unsupported input       

  Channel policy          Convert multichannel    Evaluate channel
                          recordings to model     selection/downmix on
                          input                   development data; do
                                                  not assume averaging is
                                                  best

  Log-Mel frontend        Convert waveform into   One frontend for the
                          time-frequency          custom model
                          representation          

  Convolutional stem      Extract local spectral  Compact convolution
                          patterns                

  Residual blocks         Learn reusable acoustic Depthwise-separable
                          patterns                convolutions

  Dilated blocks          Increase temporal       Dilation 1, 2, 4 as
                          receptive field without initial experiment
                          separate temporal       
                          network                 

  Latent representation   Compress learned normal Start at 64 dimensions;
                          acoustic structure      compare 128

  Decoder                 Reconstruct input       Lightweight CNN
                          representation          

  Reconstruction scoring  Measure difference      Primary score: mean
                          between input and       absolute error
                          reconstruction          

  Calibration             Interpret score         Per-machine calibration
                          relative to normal      statistics and
                          examples for a specific threshold
                          machine                 
  -----------------------------------------------------------------------

## 5. Initial engineering configuration

These are starting values to validate, not proven optimal
hyperparameters.

  Setting                   Initial value
  ------------------------- ---------------------------------
  Sample rate               16,000 Hz
  Model input               Mono float32
  Mel bins                  64 initially; compare 80
  FFT size                  1,024
  Hop length                256
  Encoder                   Compact depthwise-separable CNN
  Residual blocks           3 initial blocks
  Temporal dilation         1, 2, 4
  Latent dimension          64 initially; compare 128
  Decoder                   Lightweight CNN
  Reconstruction loss       L1
  Optimizer                 AdamW
  Learning rate             0.0005 starting point
  Maximum epochs            50
  Early stopping            Patience 8
  Product analysis window   1 second initially
  Product overlap           50% initially

All model and frontend dimensions must be tested for shape consistency.
Keep the official frontend for any future pretrained model rather than
forcing it through this custom frontend.

## 6. Training and data policy

-   Train the detector using normal recordings only.
-   Split original recording files before creating overlapping windows.
-   Fit normalization and calibration only on the permitted
    training/development normal data.
-   Do not use final-test anomalies for fitting, threshold selection, or
    model selection.
-   Keep machine-specific and cross-machine evaluation separate.
-   Maintain a manifest containing original file, machine ID, channel
    policy, split, and derived-window references.
-   Keep the original MIMII benchmark protocol distinct from the
    product's sliding-window experience.

## 7. Scoring and decision policy

For input spectrogram `X` and reconstruction `X_hat`:

``` text
reconstruction_error = mean(abs(X - X_hat))
```

The raw reconstruction error is not a probability of failure. Calibrate
it against separate normal recordings for the enrolled machine.

Initial output states:

-   **Normal-like:** valid recording and score within the calibrated
    normal range.
-   **Unusual:** valid recording and score above the chosen anomaly
    threshold.
-   **Inconclusive:** poor-quality audio, insufficient calibration,
    unsupported conditions, or an unreliable analysis.

A percentile or normalized score must be labelled as an anomaly
score/normal-reference percentile, never as "probability of machine
failure".

## 8. Explicitly excluded from the first release

-   Three independent short-, long-, and temporal-processing networks.
-   Separate attention-fusion network.
-   Separate temporal CNN branch.
-   Mahalanobis score fusion as a default production path.
-   VAE, Transformer, BEATs, PANNs, or model ensembles.
-   Exact mechanical fault diagnosis or remaining-useful-life
    prediction.

These may be reconsidered only when a controlled experiment shows a
meaningful benefit that outweighs latency, memory, reliability, and
maintenance costs.

## 9. Acceptance criteria for the architecture

Do not call the architecture successful until it:

1.  Trains reproducibly on the selected normal-only training split.
2.  Runs inference on CPU and the target GPU.
3.  Passes input-shape, audio-quality, and numerical-stability tests.
4.  Is compared with a simple baseline on untouched test recordings.
5.  Reports ROC-AUC, pAUC at max FPR 0.1, and operating-point
    false-positive/recall metrics.
6.  Reports model size, peak VRAM, CPU/GPU latency, and end-to-end
    latency.
7.  Demonstrates machine-specific calibration without test leakage.
8.  Produces interpretable results without claiming a confirmed
    mechanical fault.
