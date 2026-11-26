#!/usr/bin/env bash
# Argo Rollouts canary (argocd/canary) on a kind cluster, with Gateway API
# traffic splitting and Prometheus analysis. One script for CI and local demos.
#
#   scripts/canary.sh install   cluster add-ons (scripts/kind-addons.sh: gateway, rollouts, prometheus)
#   scripts/canary.sh test      the full lifecycle, asserting at every step:
#     v1 stable
#     v2 (good)      paused at 10 % with a 90/10 HTTPRoute split -> promote ->
#                    50 % -> Prometheus analysis passes -> 100 %, v2 stable
#     v3 (wrong)     same image, decision threshold forced to 0: fast, no errors,
#                    flags every pack "high" -> analysis fails on high-risk-share
#                    -> auto-abort, traffic snaps back to v2
#     v4 (broken)    no API in the image -> progress deadline -> auto-abort
#   scripts/canary.sh demo      install, leave v2 paused at 10 % and open the
#                               Argo Rollouts dashboard (http://localhost:3100)
#
# Env: CLUSTER (kind cluster name), SOURCE_IMAGE (a built serving image).
set -euo pipefail
shopt -s inherit_errexit # make failures inside $(...) stop the script too

CLUSTER="${CLUSTER:-ev-battery}"
SOURCE_IMAGE="${SOURCE_IMAGE:-ev-battery-api:ci}"
ROLLOUTS_VERSION="${ROLLOUTS_VERSION:-v1.10.2}"
BROKEN_IMAGE="busybox:1.36.1"           # no API inside: its pods never pass /health
LOADGEN_IMAGE="curlimages/curl:8.11.1"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/.bin"
NS=rollouts-ci
NAME=ev-battery-api
RO="rollout/$NAME"
ROUTE="httproute/$NAME"
GW_PORT=8090

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*" >&2; }
die() { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }
summary() { [ -n "${GITHUB_STEP_SUMMARY:-}" ] && echo "$*" >> "$GITHUB_STEP_SUMMARY" || true; }

# --- the kubectl-argo-rollouts CLI (downloaded to ./.bin if not on PATH) --------
rollouts_cli() {
  if command -v kubectl-argo-rollouts >/dev/null 2>&1; then echo kubectl-argo-rollouts; return; fi
  local os arch ext=""
  case "$(uname -s)" in
    Linux) os=linux ;;
    Darwin) os=darwin ;;
    MINGW* | MSYS* | CYGWIN*) os=windows; ext=".exe" ;;
    *) die "unsupported OS $(uname -s)" ;;
  esac
  case "$(uname -m)" in
    x86_64 | amd64) arch=amd64 ;;
    arm64 | aarch64) arch=arm64 ;;
    *) die "unsupported arch $(uname -m)" ;;
  esac
  # Release assets have no extension (even on Windows); save with .exe there.
  local f="$BIN/kubectl-argo-rollouts-$ROLLOUTS_VERSION$ext"
  if [ ! -x "$f" ]; then
    mkdir -p "$BIN"
    curl -sSfL -o "$f.part" \
      "https://github.com/argoproj/argo-rollouts/releases/download/$ROLLOUTS_VERSION/kubectl-argo-rollouts-$os-$arch" ||
      die "could not download kubectl-argo-rollouts $ROLLOUTS_VERSION for $os-$arch"
    mv "$f.part" "$f"
    chmod +x "$f"
  fi
  "$f" version >/dev/null || die "$f does not run"
  echo "$f"
}

install() { CLUSTER="$CLUSTER" bash "$ROOT/scripts/kind-addons.sh" gateway rollouts prometheus; }

# --- images ---------------------------------------------------------------------
# v1 and v2 are the same image under two tags (a "new version" to canary).
# v3 is that image with the decision threshold forced to 0: identical latency and
# no errors, but every prediction lands in the "high" band.
load_images() {
  docker image inspect "$SOURCE_IMAGE" >/dev/null 2>&1 ||
    die "image $SOURCE_IMAGE not found. Build it first (make docker-build)."
  log "Preparing images: ev-battery-api:ci (v1), :ci-v2 (v2), :ci-wrong (v3, threshold 0)"
  docker tag "$SOURCE_IMAGE" ev-battery-api:ci
  docker tag "$SOURCE_IMAGE" ev-battery-api:ci-v2

  local cid tmp
  tmp="$(mktemp -d)"
  cid="$(docker create ev-battery-api:ci)"
  docker cp "$cid:/app/models/model_metadata.json" "$tmp/meta.json"
  sed -E 's/"decision_threshold": *[0-9.eE+-]+/"decision_threshold": 0.0/' "$tmp/meta.json" > "$tmp/wrong.json"
  grep -q '"decision_threshold": 0.0' "$tmp/wrong.json" || die "could not patch the decision threshold"
  docker cp "$tmp/wrong.json" "$cid:/app/models/model_metadata.json"
  docker commit "$cid" ev-battery-api:ci-wrong >/dev/null
  docker rm "$cid" >/dev/null
  rm -rf "$tmp"

  kind load docker-image ev-battery-api:ci ev-battery-api:ci-v2 ev-battery-api:ci-wrong --name "$CLUSTER"
}

# --- helpers --------------------------------------------------------------------
ro_field() { kubectl get "$RO" -n "$NS" -o jsonpath="{$1}"; }
weight() { kubectl get "$ROUTE" -n "$NS" -o jsonpath="{.spec.rules[0].backendRefs[?(@.name==\"$1\")].weight}"; }

expect_weights() {
  local want_stable="$1" want_canary="$2" s="" c=""
  for _ in $(seq 1 60); do
    s="$(weight "$NAME-stable")"
    c="$(weight "$NAME-canary")"
    if [ "$s" = "$want_stable" ] && [ "$c" = "$want_canary" ]; then
      echo "HTTPRoute weights: stable=$s canary=$c" >&2
      return 0
    fi
    sleep 2
  done
  die "expected weights stable=$want_stable canary=$want_canary, got stable=$s canary=$c"
}

envoy_svc() {
  kubectl get svc -n envoy-gateway-system -o jsonpath='{.items[0].metadata.name}' \
    -l "gateway.envoyproxy.io/owning-gateway-name=ev-battery-gateway,gateway.envoyproxy.io/owning-gateway-namespace=$NS"
}

# Call the API through the Gateway (port-forward to its Envoy proxy).
gateway_health() {
  local svc pf ok=1
  svc="$(envoy_svc)"
  [ -n "$svc" ] || die "no Envoy Service found for the Gateway"
  kubectl port-forward "svc/$svc" -n envoy-gateway-system "$GW_PORT:80" >/dev/null 2>&1 &
  pf=$!
  for _ in $(seq 1 30); do
    if curl -sf "localhost:$GW_PORT/health" | grep -q '"model_loaded":true'; then ok=0; break; fi
    sleep 1
  done
  kill "$pf" 2>/dev/null || true
  [ "$ok" = 0 ] || die "API not reachable through the Gateway"
  echo "Gateway -> API /health OK" >&2
}

# ~20 req/s through the Gateway: 9 healthy packs (scored "low") for every worn
# fleet van (scored "high"), i.e. a 10 % high-risk share, inside the 5-25 % band.
start_loadgen() {
  local svc
  svc="$(envoy_svc)"
  log "Starting load generator (10 % high-risk traffic) -> $svc"
  kubectl delete pod loadgen -n "$NS" --ignore-not-found --wait >/dev/null
  kubectl apply -n "$NS" -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: loadgen
spec:
  restartPolicy: Always
  containers:
    - name: loadgen
      image: $LOADGEN_IMAGE
      env:
        - name: URL
          value: http://$svc.envoy-gateway-system.svc.cluster.local/predict
        - name: HEALTHY
          value: '{"vehicle_brand":"Hyundai","vehicle_type":"Crossover","battery_chemistry":"NMC","manufacturing_year":2022,"vehicle_age_years":2.5,"odometer_km":40000,"cycle_count":420,"state_of_health":95.0,"battery_health_percent":95.5,"internal_resistance":0.25,"cell_temperature_avg":24.0,"cell_temperature_max":33.0,"fast_charge_ratio":0.15,"overcharge_events":0,"BMS_warning_count":0,"cooling_system_health":95.0,"capacity_loss_percent":4.0,"previous_faults":0}'
        - name: WORN
          value: '{"vehicle_brand":"Nissan","vehicle_type":"Van","battery_chemistry":"LMO","manufacturing_year":2017,"vehicle_age_years":8.5,"odometer_km":330000,"cycle_count":2900,"state_of_health":62.0,"battery_health_percent":61.0,"internal_resistance":0.85,"cell_temperature_avg":41.0,"cell_temperature_max":62.0,"fast_charge_ratio":0.75,"overcharge_events":4,"BMS_warning_count":5,"cooling_system_health":70.0,"capacity_loss_percent":36.0,"previous_faults":4}'
      command: ["sh", "-c"]
      args:
        - |
          i=0
          while true; do
            i=\$((i + 1))
            if [ \$((i % 10)) -eq 0 ]; then body="\$WORN"; else body="\$HEALTHY"; fi
            curl -s -o /dev/null -X POST "\$URL" -H 'Content-Type: application/json' -d "\$body"
            sleep 0.05
          done
EOF
  kubectl wait pod/loadgen -n "$NS" --for=condition=Ready --timeout=2m
}

latest_analysisrun() {
  kubectl get analysisrun -n "$NS" --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1:].metadata.name}'
}

metric_phase() { # <analysisrun> <metric>
  kubectl get analysisrun "$1" -n "$NS" -o jsonpath="{.status.metricResults[?(@.name==\"$2\")].phase}"
}

last_value() { # <analysisrun> <metric>  -> e.g. 0.1 (Prometheus results come back as "[0.1]")
  kubectl get analysisrun "$1" -n "$NS" \
    -o jsonpath="{.status.metricResults[?(@.name==\"$2\")].measurements[-1:].value}" | tr -d '[] '
}

deploy_v1() {
  log "Deploying v1 (argocd/canary/kind overlay)"
  kubectl create namespace "$NS" --dry-run=client -o yaml | kubectl apply -f -
  kubectl delete "$RO" -n "$NS" --ignore-not-found --wait # fresh start on re-runs
  kubectl delete analysisrun --all -n "$NS" --ignore-not-found >/dev/null
  kubectl apply -k "$ROOT/argocd/canary/kind"
  # First deploy skips the canary steps and goes straight to stable.
  kubectl wait "$RO" -n "$NS" --for=jsonpath='{.status.phase}'=Healthy --timeout=3m
  kubectl wait gateway/ev-battery-gateway -n "$NS" --for=condition=Programmed --timeout=3m
  expect_weights 100 0
  gateway_health
}

# Set a new image and wait for the canary to stop at the 10 % pause.
canary_to_pause() { # <image> <label>
  log "Rolling out $2: canary must stop at 10 % with the current version still stable"
  local stable_before
  stable_before="$(ro_field .status.stableRS)"
  "$CLI" set image "$NAME" "api=$1" -n "$NS"
  kubectl wait "$RO" -n "$NS" --for=jsonpath='{.status.phase}'=Paused --timeout=3m
  [ "$(ro_field .status.currentStepIndex)" = "1" ] || die "expected to pause at step 1"
  [ "$(ro_field .status.stableRS)" = "$stable_before" ] || die "stable changed before promotion"
  expect_weights 90 10
}

# --- commands -------------------------------------------------------------------
test_canary() {
  CLI="$(rollouts_cli)"
  load_images
  deploy_v1
  start_loadgen
  local stable_v1 stable_v2 run

  # v2: good model -------------------------------------------------------------
  stable_v1="$(ro_field .status.stableRS)"
  canary_to_pause ev-battery-api:ci-v2 "v2 (good model)"
  gateway_health
  log "Promoting v2: 50 % -> Prometheus analysis must pass -> 100 %"
  "$CLI" promote "$NAME" -n "$NS"
  kubectl wait "$RO" -n "$NS" --for=jsonpath='{.status.phase}'=Healthy --timeout=5m
  "$CLI" get rollout "$NAME" -n "$NS"
  run="$(latest_analysisrun)"
  [ "$(kubectl get analysisrun "$run" -n "$NS" -o jsonpath='{.status.phase}')" = "Successful" ] ||
    die "analysis $run did not succeed for the good model"
  # Guard against a vacuous pass ("no data" counts as a pass): the last
  # high-risk-share sample must be a real number close to the 10 % we send.
  local share
  share="$(last_value "$run" high-risk-share)"
  echo "v2 high-risk share measured by Prometheus: $share" >&2
  awk -v s="$share" 'BEGIN { exit !(s + 0 == s && s >= 0.05 && s <= 0.25) }' ||
    die "v2 analysis passed without measuring a real high-risk share (got '$share')"
  stable_v2="$(ro_field .status.stableRS)"
  [ "$stable_v2" != "$stable_v1" ] || die "v2 did not become stable"
  expect_weights 100 0

  # v3: fast but wrong model ---------------------------------------------------
  canary_to_pause ev-battery-api:ci-wrong "v3 (fast but wrong: threshold 0)"
  log "Promoting v3: the high-risk-share analysis must fail and abort it"
  "$CLI" promote "$NAME" -n "$NS"
  kubectl wait "$RO" -n "$NS" --for=jsonpath='{.status.abort}'=true --timeout=5m
  "$CLI" get rollout "$NAME" -n "$NS"
  run="$(latest_analysisrun)"
  kubectl get analysisrun "$run" -n "$NS" -o jsonpath='{range .status.metricResults[*]}{.name}={.phase} {end}' >&2
  echo >&2
  [ "$(metric_phase "$run" high-risk-share)" = "Failed" ] || die "high-risk-share did not fail for the wrong model"
  echo "v3 high-risk share measured by Prometheus: $(last_value "$run" high-risk-share)" >&2
  [ "$(metric_phase "$run" error-rate)" != "Failed" ] || die "error-rate failed: v3 should only be wrong, not broken"
  [ "$(metric_phase "$run" p95-latency)" != "Failed" ] || die "p95-latency failed: v3 should only be wrong, not slow"
  [ "$(ro_field .status.stableRS)" = "$stable_v2" ] || die "stable changed after abort"
  expect_weights 100 0
  gateway_health

  # v4: broken image -----------------------------------------------------------
  log "Rolling out a broken v4 (no API): the progress deadline must abort it"
  "$CLI" set image "$NAME" "api=$BROKEN_IMAGE" -n "$NS"
  # v3 left the rollout Degraded: first see the new canary start, then the abort.
  kubectl wait "$RO" -n "$NS" --for=jsonpath='{.status.phase}'=Progressing --timeout=1m
  kubectl wait "$RO" -n "$NS" --for=jsonpath='{.status.phase}'=Degraded --timeout=5m
  kubectl wait "$RO" -n "$NS" --for=jsonpath='{.status.abort}'=true --timeout=1m
  "$CLI" get rollout "$NAME" -n "$NS"
  [ "$(ro_field .status.stableRS)" = "$stable_v2" ] || die "stable changed after abort"
  expect_weights 100 0
  gateway_health

  kubectl delete pod loadgen -n "$NS" --ignore-not-found --wait=false >/dev/null
  log "Canary test passed"
  summary "### Canary (Argo Rollouts $ROLLOUTS_VERSION, Gateway API, Prometheus analysis)"
  summary "- **v2, good model:** paused at 10 % (HTTPRoute 90/10, v1 still stable) → promoted → analysis **passed** (latency, error rate, high-risk share = $share) → 100 %"
  summary "- **v3, fast but wrong** (threshold 0, every pack \"high\"): latency and errors fine, **high-risk-share failed** → auto-aborted, route back to 100 % v2"
  summary "- **v4, broken image:** never ready → **progress deadline aborted it**, v2 kept serving"
}

demo() {
  install
  CLI="$(rollouts_cli)"
  load_images
  deploy_v1
  start_loadgen
  canary_to_pause ev-battery-api:ci-v2 "v2"
  cat >&2 <<EOF

Canary is paused at 10 % (HTTPRoute: stable 90 / canary 10), with traffic flowing.
Dashboard: http://localhost:3100/rollouts/rollout/$NS/$NAME
  - screenshot it, then click Promote: 50 % -> Prometheus analysis -> 100 %
  - try the wrong model:  $CLI set image $NAME api=ev-battery-api:ci-wrong -n $NS   (then Promote)
  - try a broken image:   $CLI set image $NAME api=$BROKEN_IMAGE -n $NS
  - watch in a terminal:  $CLI get rollout $NAME -n $NS --watch
Ctrl+C stops the dashboard; the cluster keeps running (make kind-down to delete it).
EOF
  "$CLI" dashboard
}

case "${1:-}" in
  install) install ;;
  test) test_canary ;;
  demo) demo ;;
  *) die "usage: $0 install|test|demo" ;;
esac
