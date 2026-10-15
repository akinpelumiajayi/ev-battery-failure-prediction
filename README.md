# EV Battery Failure Prediction — End-to-End MLOps Pipeline

A production-style MLOps system that predicts EV battery failure from
telemetry/usage data, covering the full lifecycle: **data validation →
versioning → experiment tracking → CI → containerization → Kubernetes
deployment → GitOps delivery → monitoring → drift-triggered retraining.**

Trained on a 20,000-vehicle dataset (70 features: battery chemistry,
charging behavior, driving patterns, thermal/electrical sensor readings).
Held-out test performance: **ROC-AUC 0.956, F1 0.594** (imbalanced target,
~6.9% failure rate) with a RandomForest baseline.

## Architecture

```
┌─────────────┐   ┌──────────────┐   ┌─────────────┐   ┌───────────┐
│ Raw CSV     │──▶│ Validate     │──▶│ Preprocess   │──▶│ Featurize │
│ (DVC-tracked)│   │ (pandera)    │   │ (split)      │   │ (sklearn) │
└─────────────┘   └──────────────┘   └─────────────┘   └─────┬─────┘
                                                               │
        ┌──────────────────────────────────────────────────────┘
        ▼
┌─────────────┐   ┌──────────────┐   ┌─────────────────┐
│ Train        │──▶│ Evaluate      │──▶│ Quality gate     │
│ (MLflow log/ │   │ (test metrics)│   │ (blocks bad model)│
│  registry)   │   └──────────────┘   └────────┬─────────┘
└─────────────┘                                 │
                                                 ▼
┌────────────┐  build   ┌───────────┐  push   ┌────────────────┐
│ CI (GitHub  │────────▶│ Docker     │───────▶│ GHCR registry   │
│  Actions)   │          │ image      │        └────────┬───────┘
└────────────┘                                            │
                                                            ▼
┌────────────┐  commit  ┌────────────┐  watches ┌────────────────┐
│ CD workflow │────────▶│ Git (Helm   │◀────────│ ArgoCD          │
│ (bump tag)  │          │ values)    │  sync    │ (staging: auto, │
└────────────┘                                    │  prod: manual)  │
                                                    └────────┬────────┘
                                                              ▼
                                              ┌───────────────────────┐
                                              │ Kubernetes (HPA,       │
                                              │ FastAPI pods, Ingress) │
                                              └───────────┬───────────┘
                                                            ▼
                              ┌───────────────────────────────────────┐
                              │ Prometheus + Grafana (latency, volume, │
                              │ risk-band distribution) + drift check  │
                              │ (scheduled) ──▶ opens issue / triggers │
                              │ retraining, closing the loop           │
                              └───────────────────────────────────────┘
```

## Repo layout

```
├── data/                # raw (DVC-tracked), processed, reference snapshot
├── src/
│   ├── data/             # validate.py, preprocess.py
│   ├── features/         # build_features.py — sklearn ColumnTransformer
│   ├── training/         # train.py — MLflow tracking + registry
│   ├── evaluation/        # evaluate.py — test metrics + quality gate
│   ├── serving/           # app.py (FastAPI), schemas.py
│   └── monitoring/        # drift_check.py — PSI-based drift detection
├── tests/                # pytest: data validation, features, API
├── dvc.yaml, params.yaml # DVC pipeline DAG + all hyperparameters/config
├── docker/                # Dockerfile.train, Dockerfile.serve
├── docker-compose.yml     # local stack: MLflow, API, Prometheus, Grafana
├── k8s/base/               # plain Kubernetes manifests (Deployment/HPA/Ingress)
├── helm/ev-battery-model/  # Helm chart (what ArgoCD actually deploys)
├── argocd/                 # ArgoCD Application manifests (staging + prod)
│   └── rollout-canary-example.yaml   # optional Argo Rollouts canary strategy
├── monitoring/              # Prometheus config, alert rules, Grafana dashboard
└── .github/workflows/       # ci.yml, cd.yml, drift-check.yml
```

## Quickstart (local)

```bash
python -m venv .venv && source .venv/bin/activate
make setup

# Run the full pipeline: validate -> preprocess -> featurize -> train -> evaluate -> drift_check
dvc repro

# Inspect metrics / DAG
dvc metrics show
dvc dag

# Push versioned data + model artifacts to the local DVC remote (./dvcstore)
make dvc-push

# Serve the model
make serve
curl localhost:8000/health
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"vehicle_brand": "Tesla", "battery_health_percent": 55.0, "state_of_health": 54.0}'

# Run the whole local stack (MLflow UI + API + Prometheus + Grafana)
make docker-up
```

## DVC — local storage

This project uses a **local filesystem DVC remote** (`./dvcstore`) so it
runs with zero cloud credentials. It's configured in `.dvc/config`:

```ini
[core]
    remote = localstorage
['remote "localstorage"']
    url = ../dvcstore
```

`data/raw/ev_battery_data.csv` is tracked via `data/raw/ev_battery_data.csv.dvc`
(the actual CSV is git-ignored — only its hash pointer is committed).
`dvc push` copies content-addressed blobs into `dvcstore/`; `dvc pull`
restores them on a fresh checkout or in CI. **To point this at a real
team/cloud remote later**, it's a one-line change:

```bash
dvc remote add -d cloud s3://your-bucket/dvcstore   # or gs://, azure://
dvc push -r cloud
```

## MLflow

Local tracking store (`sqlite:///mlflow/mlflow.db`), local artifact store
(`./mlflow/mlruns`). Every `train.py` run logs params, validation metrics,
and registers a model version under `ev-battery-failure-model`.

```bash
make mlflow-ui   # http://localhost:5000
```

Swap `tracking_uri` in `params.yaml` for a hosted MLflow server
(`http://mlflow.internal:5000`) to share the registry across a team — no
other code changes needed.

## CI/CD — GitOps with ArgoCD

1. **CI** (`ci.yml`): on every push to `main` — pulls data via DVC, runs
   the full `dvc repro` pipeline, runs tests, enforces the ROC-AUC quality
   gate, pushes updated artifacts to the DVC remote, then builds and pushes
   a Docker image to GHCR.
2. **CD** (`cd.yml`): on CI success, bumps `image.tag` in
   `helm/ev-battery-model/values-staging.yaml` and commits it back to
   `main`. **No `kubectl` or `helm` command runs in CI** — Git is the only
   thing that changes.
3. **ArgoCD** (`argocd/application-staging.yaml`) watches that same path
   in Git and auto-syncs (`prune: true, selfHeal: true`) — any manual
   cluster drift gets reverted back to what's in Git.
4. **Production** (`argocd/application-prod.yaml`) has no `automated:`
   sync policy — promotion requires an explicit `argocd app sync` after
   staging validation, and `values-prod.yaml` pins an immutable image tag
   rather than `latest`.
5. **Progressive delivery**: `argocd/rollout-canary-example.yaml` shows how
   to swap the plain Deployment for an Argo Rollouts canary that shifts
   traffic gradually and auto-analyzes Prometheus latency before
   proceeding.

## Monitoring & closing the loop

- `/metrics` on the FastAPI app exposes prediction volume by risk band,
  latency histograms, and the predicted-probability distribution —
  scraped by Prometheus (`monitoring/prometheus.yml`), visualized in
  Grafana (`monitoring/grafana-dashboard.json`), alerted on
  (`monitoring/alert_rules.yml`).
- `src/monitoring/drift_check.py` computes PSI per feature against the
  training-set reference snapshot. `.github/workflows/drift-check.yml`
  runs this daily; if drift crosses `params.yaml`'s
  `monitoring.drift_psi_threshold`, it opens a GitHub issue — wire that
  same signal to re-trigger `ci.yml` (`workflow_dispatch` /
  `repository_dispatch`) to fully automate retraining.

## Running on Kubernetes

```bash
# Local cluster for testing (kind or minikube)
kind create cluster

# Install ArgoCD, then bootstrap the two Applications:
make argocd-bootstrap

# Or install the Helm chart directly without ArgoCD:
make helm-install-staging
kubectl port-forward svc/ev-battery-staging-ev-battery-api -n mlops-staging 8000:80
```

## Design notes / what to highlight in an interview

- **Train/serve skew prevention**: the exact fitted `ColumnTransformer`
  (imputer + scaler + encoder) is persisted and reused by the API — the
  API never re-derives feature logic.
- **Quality gates, not just metrics**: `evaluate.py` exits non-zero below
  a ROC-AUC threshold, so CI structurally cannot ship a regressed model.
- **GitOps over push-based deploys**: CI never touches the cluster
  directly; it only edits Git, which is the single source of truth and
  gives a full audit trail of every production change.
- **Staging-auto / prod-manual**: a deliberate safety asymmetry — fast
  iteration in staging, a human gate before customer-facing traffic.
- **Local-first, cloud-portable**: DVC remote and MLflow backend are both
  local by default (zero cloud cost to run this portfolio project) but
  are one config line away from S3/GCS + a hosted MLflow server.
