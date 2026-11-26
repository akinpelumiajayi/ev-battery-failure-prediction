#!/usr/bin/env bash
# Platform add-ons a production cluster would already have, installed on kind
# so CI and local runs can exercise the real manifests against them.
#
#   scripts/kind-addons.sh [gateway] [rollouts] [prometheus] [metrics]   (default: all)
#
#   gateway     Envoy Gateway (bundles the Gateway API CRDs), GatewayClass "eg",
#               and the shared Gateway "public" in gateway-system (prod's parentRef)
#   rollouts    Argo Rollouts controller + the Gateway API traffic-router plugin
#   prometheus  kube-prometheus-stack, slimmed to Prometheus + its operator
#               (Prometheus at prometheus-operated.monitoring.svc:9090)
#   metrics     metrics-server, so HorizontalPodAutoscalers report Healthy
#
# Idempotent. Env: CLUSTER (kind cluster name, default ev-battery).
set -euo pipefail

CLUSTER="${CLUSTER:-ev-battery}"
ENVOY_GATEWAY_VERSION="${ENVOY_GATEWAY_VERSION:-v1.10.0}"
ROLLOUTS_VERSION="${ROLLOUTS_VERSION:-v1.10.2}"
GATEWAYAPI_PLUGIN_VERSION="${GATEWAYAPI_PLUGIN_VERSION:-v0.17.0}"
KUBE_PROMETHEUS_STACK_VERSION="${KUBE_PROMETHEUS_STACK_VERSION:-92.22.0}"
METRICS_SERVER_VERSION="${METRICS_SERVER_VERSION:-v0.9.1}"

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*" >&2; }

gateway() {
  log "Envoy Gateway $ENVOY_GATEWAY_VERSION (bundles the Gateway API CRDs)"
  helm upgrade --install eg oci://docker.io/envoyproxy/gateway-helm --version "$ENVOY_GATEWAY_VERSION" \
    -n envoy-gateway-system --create-namespace --wait --timeout 5m
  # kind has no LoadBalancer: run the Envoy proxies behind ClusterIP Services so
  # Gateways get an address and report Programmed.
  kubectl apply -f - <<'EOF'
apiVersion: gateway.envoyproxy.io/v1alpha1
kind: EnvoyProxy
metadata:
  name: clusterip
  namespace: envoy-gateway-system
spec:
  provider:
    type: Kubernetes
    kubernetes:
      envoyService:
        type: ClusterIP
---
apiVersion: gateway.networking.k8s.io/v1
kind: GatewayClass
metadata:
  name: eg
spec:
  controllerName: gateway.envoyproxy.io/gatewayclass-controller
  parametersRef:
    group: gateway.envoyproxy.io
    kind: EnvoyProxy
    name: clusterip
    namespace: envoy-gateway-system
---
apiVersion: v1
kind: Namespace
metadata:
  name: gateway-system
---
# The cluster's shared entry point. values-prod.yaml attaches its HTTPRoute here.
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: public
  namespace: gateway-system
spec:
  gatewayClassName: eg
  listeners:
    - name: http
      protocol: HTTP
      port: 80
      allowedRoutes:
        namespaces:
          from: All
EOF
  kubectl wait gatewayclass/eg --for=condition=Accepted --timeout=2m
  kubectl wait gateway/public -n gateway-system --for=condition=Programmed --timeout=3m
}

rollouts() {
  log "Argo Rollouts $ROLLOUTS_VERSION + Gateway API plugin $GATEWAYAPI_PLUGIN_VERSION"
  kubectl create namespace argo-rollouts --dry-run=client -o yaml | kubectl apply -f -
  # Server-side apply: the Rollout and AnalysisRun CRDs are too big for the
  # last-applied-configuration annotation that client-side apply writes (256 KB).
  kubectl apply -n argo-rollouts --server-side --force-conflicts \
    -f "https://github.com/argoproj/argo-rollouts/releases/download/$ROLLOUTS_VERSION/install.yaml"
  local arch
  arch="$(kubectl get nodes -o jsonpath='{.items[0].status.nodeInfo.architecture}')"
  kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: argo-rollouts-config
  namespace: argo-rollouts
data:
  trafficRouterPlugins: |-
    - name: "argoproj-labs/gatewayAPI"
      location: "https://github.com/argoproj-labs/rollouts-plugin-trafficrouter-gatewayapi/releases/download/$GATEWAYAPI_PLUGIN_VERSION/gatewayapi-plugin-linux-$arch"
---
# Least privilege for the plugin: manage routes, read gateways, keep its state ConfigMap.
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: argo-rollouts-gatewayapi
rules:
  - apiGroups: ["gateway.networking.k8s.io"]
    resources: ["httproutes", "grpcroutes", "tcproutes"]
    verbs: ["get", "list", "watch", "update", "patch"]
  - apiGroups: ["gateway.networking.k8s.io"]
    resources: ["gateways"]
    verbs: ["get", "list", "watch"]
  - apiGroups: [""]
    resources: ["configmaps"]
    verbs: ["get", "list", "watch", "create", "update", "patch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: argo-rollouts-gatewayapi
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: argo-rollouts-gatewayapi
subjects:
  - kind: ServiceAccount
    name: argo-rollouts
    namespace: argo-rollouts
EOF
  # The controller loads traffic-router plugins at start-up.
  kubectl rollout restart deployment/argo-rollouts -n argo-rollouts
  kubectl rollout status deployment/argo-rollouts -n argo-rollouts --timeout=3m
}

prometheus() {
  log "kube-prometheus-stack $KUBE_PROMETHEUS_STACK_VERSION (Prometheus + operator only)"
  helm repo add prometheus-community https://prometheus-community.github.io/helm-charts --force-update >/dev/null
  # Release name "kube-prometheus-stack" matches the release label the chart's
  # ServiceMonitor and the canary's PodMonitor carry; the selectors below also
  # pick up monitors without it.
  helm upgrade --install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
    --version "$KUBE_PROMETHEUS_STACK_VERSION" -n monitoring --create-namespace \
    --set grafana.enabled=false \
    --set alertmanager.enabled=false \
    --set nodeExporter.enabled=false \
    --set kubeStateMetrics.enabled=false \
    --set defaultRules.create=false \
    --set kubeEtcd.enabled=false --set kubeScheduler.enabled=false \
    --set kubeControllerManager.enabled=false --set kubeProxy.enabled=false \
    --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
    --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false \
    --set prometheus.prometheusSpec.ruleSelectorNilUsesHelmValues=false \
    --set prometheus.prometheusSpec.scrapeInterval=15s \
    --wait --timeout 8m
  kubectl wait pod -n monitoring -l app.kubernetes.io/name=prometheus --for=condition=Ready --timeout=5m
}

metrics() {
  log "metrics-server $METRICS_SERVER_VERSION"
  kubectl apply -f "https://github.com/kubernetes-sigs/metrics-server/releases/download/$METRICS_SERVER_VERSION/components.yaml"
  # kind's kubelets use self-signed serving certs.
  kubectl patch deployment metrics-server -n kube-system --type=json \
    -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]' \
    2>/dev/null || true
  kubectl rollout status deployment/metrics-server -n kube-system --timeout=3m
}

kubectl config use-context "kind-$CLUSTER" >/dev/null
[ $# -gt 0 ] || set -- gateway rollouts prometheus metrics
for addon in "$@"; do
  case "$addon" in
    gateway | rollouts | prometheus | metrics) "$addon" ;;
    *) echo "unknown add-on: $addon" >&2; exit 1 ;;
  esac
done
