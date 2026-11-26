# Run `make help` for a list of targets.
PY ?= python
IMAGE ?= ev-battery-api:local
.DEFAULT_GOAL := help

.PHONY: help
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---- setup ---------------------------------------------------------------------
.PHONY: setup lock
setup:  ## Install dev dependencies + pre-commit hooks
	$(PY) -m pip install -r requirements/dev.txt
	pre-commit install

lock:  ## Re-compile pinned requirements/*.txt from *.in (needs uv)
	cd requirements && for f in serve train dev demo; do \
		uv pip compile $$f.in --python-version 3.12 --python-platform x86_64-manylinux_2_28 -q -o $$f.txt; done

# ---- ML pipeline ---------------------------------------------------------------
.PHONY: data pipeline metrics dag
data:  ## Download + md5-verify the raw dataset
	$(PY) -m src.data.fetch_data

pipeline: data  ## Run the full DVC pipeline (validate -> ... -> evaluate -> drift)
	dvc repro

metrics:  ## Show tracked metrics (and diff vs. main)
	dvc metrics show
	dvc metrics diff main --md || true

dag:  ## Print the pipeline DAG
	dvc dag

# ---- quality -------------------------------------------------------------------
.PHONY: lint format typecheck test check
lint:  ## Ruff lint + format check
	ruff check src tests demo scripts
	ruff format --check src tests demo scripts

format:  ## Auto-format
	ruff format src tests demo scripts
	ruff check --fix src tests demo scripts

typecheck:  ## mypy
	mypy

test:  ## Unit + API tests with coverage
	pytest --cov --cov-report=term

check: lint typecheck test  ## Everything CI checks before training

# ---- drift demo ----------------------------------------------------------------
.PHONY: drift-check drift-demo drift-production
drift-check:  ## PSI drift check of the held-out test batch (expect: no drift)
	$(PY) -m src.monitoring.drift_check --current data/processed/test.parquet

drift-production:  ## Drift check on the traffic the local stack logged (after docker-up + load-test; logs flush every 30 s)
	rm -rf data/inference_logs
	docker compose cp api:/var/lib/ev-battery/inference-logs data/inference_logs
	$(PY) -m src.monitoring.inference_logs --uri data/inference_logs --min-rows 100 --out data/drift/production.parquet
	$(PY) -m src.monitoring.drift_check --current data/drift/production.parquet --report artifacts/drift_report_production.json

drift-demo:  ## Simulate a heatwave batch and watch the drift alarm fire
	$(PY) -m src.monitoring.simulate_drift --scenario heatwave
	$(PY) -m src.monitoring.drift_check --current data/drift/heatwave.parquet --report artifacts/drift_report_heatwave.json

# ---- serving -------------------------------------------------------------------
.PHONY: serve demo mlflow-ui load-test
serve:  ## Run the API locally with hot reload
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

demo:  ## Run the Streamlit UI against the local API
	API_URL=http://localhost:8000 streamlit run demo/streamlit_app.py

mlflow-ui:  ## Browse experiments + model registry
	mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --port 5000

load-test:  ## Send synthetic traffic to the API for 60s
	$(PY) scripts/load_test.py --url http://localhost:8000 --rps 10 --duration 60

# ---- containers ----------------------------------------------------------------
.PHONY: docker-build docker-up docker-down
docker-build:  ## Build serving + training images
	docker build -f docker/Dockerfile.serve -t $(IMAGE) .
	docker build -f docker/Dockerfile.train -t ev-battery-train:local .

docker-up:  ## Local stack: API, demo UI, MLflow, Prometheus, Grafana
	docker compose up --build -d
	@echo "Demo http://localhost:8501 | API http://localhost:8000/docs | Grafana http://localhost:3000"

docker-down:  ## Stop the local stack
	docker compose down -v

# ---- Kubernetes ----------------------------------------------------------------
.PHONY: kind-up kind-deploy kind-down helm-lint argocd-bootstrap kind-addons canary-test canary-demo
helm-lint:  ## Lint the Helm chart for every environment
	helm lint helm/ev-battery-model -f helm/ev-battery-model/values-staging.yaml
	helm lint helm/ev-battery-model -f helm/ev-battery-model/values-prod.yaml

kind-up:  ## Create a local kind cluster
	kind create cluster --name ev-battery

kind-deploy: docker-build  ## Build, load into kind and install the chart (staging values)
	kind load docker-image $(IMAGE) --name ev-battery
	helm upgrade --install ev-battery helm/ev-battery-model \
		-f helm/ev-battery-model/values-staging.yaml \
		--set image.repository=ev-battery-api --set image.tag=local --set image.pullPolicy=Never \
		-n mlops-staging --create-namespace --wait --timeout 3m
	@echo "kubectl port-forward svc/ev-battery-ev-battery-api -n mlops-staging 8000:80"

kind-down:  ## Delete the kind cluster
	kind delete cluster --name ev-battery

argocd-bootstrap:  ## Register the staging + prod ArgoCD Applications
	kubectl apply -f argocd/application-staging.yaml
	kubectl apply -f argocd/application-prod.yaml

kind-addons:  ## Install Gateway API (Envoy Gateway), Argo Rollouts, Prometheus and metrics-server on kind
	CLUSTER=ev-battery bash scripts/kind-addons.sh

canary-test:  ## Canary on kind, same as CI: 90/10 split, analysis passes good / aborts wrong + broken (needs kind-up + docker-build)
	CLUSTER=ev-battery SOURCE_IMAGE=$(IMAGE) bash scripts/canary.sh install
	CLUSTER=ev-battery SOURCE_IMAGE=$(IMAGE) bash scripts/canary.sh test

canary-demo:  ## Canary paused at 10 % + Argo Rollouts dashboard on http://localhost:3100 (needs kind-up + docker-build)
	CLUSTER=ev-battery SOURCE_IMAGE=$(IMAGE) bash scripts/canary.sh demo

# ---- housekeeping --------------------------------------------------------------
.PHONY: clean
clean:  ## Remove generated data, models and caches (keeps raw data)
	rm -rf data/processed data/reference data/drift models/*.joblib artifacts/*.joblib \
	       mlflow/mlflow.db mlruns .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
