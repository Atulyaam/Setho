# Stetho --- Application and Deployment Architecture

**Goal:** A deployed, judge-accessible application that demonstrates the
complete machine enrollment and acoustic inspection workflow.

## 1. Deployment-aware system diagram

``` mermaid
flowchart TD
    J[Hackathon Judge / User] --> HTTPS[HTTPS]
    HTTPS --> FE[Deployed Next.js Frontend]
    FE --> API[Deployed FastAPI Backend]
    API --> DB[(Managed PostgreSQL)]
    API --> STORE[(Private Object Storage)]
    API --> Q[Analysis Job / Worker]
    Q --> AUDIO[Audio Validation and Preprocessing]
    AUDIO --> MODEL[Versioned Stetho-MSAE Artifact]
    MODEL --> CAL[Machine Calibration Profile]
    CAL --> RESULT[Result and Evidence]
    RESULT --> DB
    RESULT --> FE
```

## 2. Application responsibilities

  -----------------------------------------------------------------------
  Layer                               Responsibilities
  ----------------------------------- -----------------------------------
  Next.js frontend                    Landing page, dashboard, machine
                                      enrollment, recording/upload,
                                      result visualizations, report view

  FastAPI backend                     API validation,
                                      authentication/authorization,
                                      profile management, job
                                      orchestration, result retrieval

  ML inference module                 Audio preprocessing, model loading,
                                      inference, reconstruction error,
                                      calibration

  PostgreSQL                          Users, machine profiles, enrollment
                                      metadata, jobs, results, model
                                      version references

  Object storage                      Original audio, optional processed
                                      artifacts, exported reports

  Worker                              Long-running audio analysis without
                                      holding a browser request open
  -----------------------------------------------------------------------

Keep the ML inference code importable and testable independently from
FastAPI. The API should call a stable inference interface rather than
contain model logic in route handlers.

## 3. Suggested deployment topology

Use the simplest provider combination that supports the project's actual
resource and pricing constraints:

-   Frontend: a managed Next.js hosting service.
-   Backend and worker: a container-capable Python hosting service.
-   Database: managed PostgreSQL.
-   Audio storage: private S3-compatible object storage.
-   Model artifact: versioned artifact stored in object storage or
    included in a controlled container image.

Do not assume free hosting includes a GPU. The first deployment must
support CPU inference. If the target host cannot reliably run the model
within the demo latency budget, use a separately hosted inference worker
or a controlled live-demo machine with a clearly documented fallback.

## 4. API outline

  ------------------------------------------------------------------------------------------
  Method                  Endpoint                                   Purpose
  ----------------------- ------------------------------------------ -----------------------
  POST                    `/api/machines`                            Create a machine
                                                                     profile

  GET                     `/api/machines`                            List machine profiles

  GET                     `/api/machines/{machine_id}`               Get profile and
                                                                     enrollment status

  POST                    `/api/machines/{machine_id}/enrollments`   Upload/register normal
                                                                     enrollment audio

  POST                    `/api/machines/{machine_id}/analyses`      Submit a new audio
                                                                     analysis

  GET                     `/api/analyses/{analysis_id}`              Retrieve job status and
                                                                     result

  GET                     `/api/analyses/{analysis_id}/report`       Retrieve inspection
                                                                     report

  GET                     `/health`                                  Liveness check

  GET                     `/ready`                                   Readiness check,
                                                                     including model
                                                                     availability
  ------------------------------------------------------------------------------------------

Use asynchronous job status for longer recordings. Do not keep a request
open indefinitely while processing audio.

## 5. Model artifact and reproducibility

Each deployed model artifact must have:

-   Model version and architecture version.
-   Training dataset manifest hash.
-   Preprocessing configuration/version.
-   Machine type and machine IDs represented during training.
-   Training split and random seed.
-   Calibration profile version.
-   Evaluation metrics and evaluation split.
-   Git commit or release identifier.

Never silently change the deployed model or its calibration profile. A
model and its matching preprocessing/calibration configuration should be
deployed as one compatible version.

## 6. Security and reliability

-   Store secrets in hosting environment variables, never in Git.
-   Use HTTPS.
-   Enforce upload size, duration, format, and content validation.
-   Store uploaded audio privately; use short-lived signed URLs when
    needed.
-   Apply authentication and authorization to machine and recording
    data.
-   Add rate limits and sensible request timeouts.
-   Do not expose raw filesystem paths or unrestricted storage buckets.
-   Log request/job IDs, model version, processing duration, and errors;
    avoid logging sensitive audio contents.
-   Provide CPU fallback and clear failure states.
-   Add database backups and a deployment rollback path.

## 7. Deployment acceptance checklist

-   [ ] Frontend is reachable over HTTPS.
-   [ ] Backend health and readiness endpoints work.
-   [ ] Database persists profiles and results across restarts.
-   [ ] Audio files persist in private object storage.
-   [ ] Model artifact loads successfully after a fresh deployment.
-   [ ] Enrollment works from the deployed frontend.
-   [ ] Analysis jobs complete and results can be retrieved.
-   [ ] CPU inference works on the chosen host.
-   [ ] Upload validation and failure states work.
-   [ ] Secrets are not committed to the repository.
-   [ ] A fresh user/browser can complete the demo workflow.
-   [ ] A fallback demo path is documented and labelled honestly.
