.PHONY: setup validate pipeline test serve docker-build docker-up docker-down \
        dvc-push dvc-pull drift-check clean lint

setup:
	pip install -r requirements.txt

validate:
	python src/data/validate.py

pipeline:
	dvc repro

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

serve:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --port 5000

drift-check:
	python src/monitoring/drift_check.py --current data/processed/test.parquet

dvc-push:
	dvc push

dvc-pull:
	dvc pull

docker-build:
	docker build -f docker/Dockerfile.serve -t ev-battery-api:local .
	docker build -f docker/Dockerfile.train -t ev-battery-train:local .

docker-up:
	docker compose up --build

docker-down:
	docker compose down -v

# Kubernetes / Helm (requires a cluster context set, e.g. via `kind` or minikube)
helm-lint:
	helm lint helm/ev-battery-model

helm-install-staging:
	helm upgrade --install ev-battery-staging helm/ev-battery-model \
		-f helm/ev-battery-model/values.yaml \
		-f helm/ev-battery-model/values-staging.yaml \
		-n mlops-staging --create-namespace

argocd-bootstrap:
	kubectl apply -f argocd/application-staging.yaml
	kubectl apply -f argocd/application-prod.yaml

clean:
	rm -rf data/processed/* models/*.joblib models/*.json artifacts/*.joblib \
	       artifacts/*.json artifacts/*.txt mlflow/mlflow.db mlflow/mlruns __pycache__ .pytest_cache
