# Stetho --- Sprint Roadmap and Repository Plan

## 1. Sprint execution rules

Every sprint must finish with:

1.  Working code.
2.  Automated or repeatable tests.
3.  Recorded evidence/results.
4.  Updated documentation.
5.  A deployment impact check.

Do not start a new architecture experiment or add a dependency without a
concrete requirement.

## 2. Sprint roadmap

``` mermaid
flowchart TD
    S0[Sprint 0: Foundation and deployment skeleton] --> S1[Sprint 1: Dataset and audio pipeline]
    S1 --> S2[Sprint 2: Baseline detector]
    S2 --> S3[Sprint 3: Stetho-MSAE]
    S3 --> S4[Sprint 4: Evaluation and calibration]
    S4 --> S5[Sprint 5: FastAPI and persistence]
    S5 --> S6[Sprint 6: Next.js product experience]
    S6 --> S7[Sprint 7: Integration and hardening]
    S7 --> S8[Sprint 8: Public demo and submission]
```

  --------------------------------------------------------------------------------
  Sprint                  Main work                        Exit condition
  ----------------------- -------------------------------- -----------------------
  0                       Repo structure, Python           App skeleton starts
                          environment, frontend/backend    locally; health
                          skeleton, Docker, environment    endpoints work
                          configuration                    

  1                       MIMII inspection, manifests,     Reproducible audio
                          audio validation, channel        pipeline and tests
                          policy, preprocessing            

  2                       Simple baseline,                 Baseline metrics and
                          training/evaluation scripts      artifacts recorded

  3                       Shared multi-scale               Model trains and runs
                          encoder-decoder                  CPU/GPU inference

  4                       Machine-specific/cross-machine   Leakage-checked
                          protocols, calibration,          evaluation report
                          ablations                        

  5                       FastAPI, PostgreSQL, private     API integration tests
                          audio storage, analysis jobs     pass

  6                       Next.js enrollment,              Complete user workflow
                          recording/upload, result views,  works locally
                          reports                          

  7                       Container deployment, latency    Staging workflow passes
                          optimization, security, failure  end-to-end
                          handling                         

  8                       Public release, demo rehearsal,  Judge can complete the
                          README, Kiro evidence            demo from a clean
                                                           session
  --------------------------------------------------------------------------------

## 3. Suggested repository structure

``` text
stetho/
├── initial.md
├── README.md
├── .gitignore
├── .env.example
├── docker-compose.yml
├── apps/
│   └── web/                      # Next.js frontend
├── services/
│   └── api/                      # FastAPI application
├── ml/
│   ├── configs/
│   ├── data/
│   │   ├── manifests/
│   │   └── splits/
│   ├── preprocessing/
│   ├── models/
│   │   └── stetho_msae/
│   ├── training/
│   ├── scoring/
│   ├── calibration/
│   ├── evaluation/
│   └── artifacts/                # Keep large artifacts out of Git
├── packages/
│   └── stetho_audio/             # Shared audio validation/preprocessing
├── infra/
│   ├── Dockerfiles/
│   └── deployment/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── docs/
│   ├── architecture/
│   ├── evaluation/
│   ├── deployment/
│   └── demo/
└── .kiro/
    ├── steering/
    ├── specs/
    └── hooks/
```

Create only the directories needed for the current sprint. Do not create
empty modules just to match the proposed tree. Dataset files, secrets,
and large model artifacts should not be committed to Git.

## 4. Sprint 0 first tasks

1.  Inspect and preserve the existing `initial.md`.
2.  Check installed Python, Node.js, Git, Docker, CUDA, and PyTorch.
3.  Create the minimal repo skeleton.
4.  Add `.gitignore` and `.env.example`.
5.  Create a minimal FastAPI health/readiness service.
6.  Create a minimal Next.js page.
7.  Add a local Docker Compose setup only for services needed
    immediately.
8.  Record exact environment versions and startup commands.
9.  Verify a clean local startup.

## 5. Definition of done for the project

The project is not complete merely because the model trains. It is
complete when the deployed application supports enrollment, analysis,
calibrated results, useful visual evidence, clear limitations,
persistent data, error handling, and a rehearsed fallback demo.
