#!/usr/bin/env bash
# Deploy a small Python web app to the team cluster at /app/longtail.
#
# Usage:
#   ./deploy.sh <app-dir> [extra-file ...]     e.g. ./deploy.sh web longtail.py
#   APP_NAME=longtail-search ./deploy.sh ./other-app
#
# The app directory must contain main.py that listens on 0.0.0.0:$PORT and serves
# its pages at / plus /health (the Ingress strips the /app/longtail prefix), and may
# contain requirements.txt. Extra files are copied next to main.py (subdirectories
# are not). Links in its HTML must be relative so they stay under /app/longtail. Only
# /app and paths below it are forwarded by the public host.
#
# Reads INGRESS_URL, USERNAME and PASSWORD from the environment, plus WANDB_API_KEY,
# WANDB_TEAM and WANDB_PROJECT when set. All of them reach the app as environment
# variables (VSS_URL, VSS_USERNAME, VSS_PASSWORD, WANDB_*).
set -euo pipefail

APP_DIR="${1:?usage: ./deploy.sh <app-dir> [extra-file ...]}"
shift
EXTRA_FILES=("$@")
APP_NAME="${APP_NAME:-longtail}"
APP_PATH="/app/longtail"
APP_PORT=8080
PUBLIC_HOST="${PUBLIC_HOST:-team-1-app.thecosmoslabs.com}"
VSS_IN_CLUSTER="http://video-backend-service:8000"

[[ -f "$APP_DIR/main.py" ]] || { echo "no main.py in $APP_DIR" >&2; exit 1; }
for f in "${EXTRA_FILES[@]}"; do
  [[ -f "$f" ]] || { echo "extra file not found: $f" >&2; exit 1; }
done
[[ "$APP_NAME" == longtail || "$APP_NAME" == longtail-* ]] || { echo "APP_NAME must be longtail or start with longtail-" >&2; exit 1; }
for v in INGRESS_URL USERNAME PASSWORD; do
  [[ -n "${!v:-}" ]] || { echo "missing environment variable $v" >&2; exit 1; }
done

export PATH="$HOME/.local/bin:$PATH"
command -v kubectl >/dev/null || { echo "kubectl not found" >&2; exit 1; }
if [[ -z "${KUBECONFIG:-}" ]]; then
  shopt -s nullglob
  configs=(/config/*-k8s.yaml)
  (( ${#configs[@]} == 1 )) || { echo "set KUBECONFIG (expected one /config/*-k8s.yaml)" >&2; exit 1; }
  export KUBECONFIG="${configs[0]}"
fi

NS="${NS:-$USERNAME}"
APP_HOST="${INGRESS_URL#*://}"
APP_HOST="${APP_HOST%%/*}"

owner=$(kubectl -n "$NS" get ingress -o jsonpath='{range .items[*]}{.metadata.name}{" "}{range .spec.rules[*]}{.host}{" "}{range .http.paths[*]}{.path}{" "}{end}{end}{"\n"}{end}' \
  | awk -v host="$APP_HOST" -v p="$APP_PATH(/|\$)(.*)" -v me="$APP_NAME" \
      '{ for (i = 3; i <= NF; i++) if ($2 == host && $i == p && $1 != me) print $1 }')
if [[ -n "$owner" ]]; then
  echo "$APP_PATH is already routed to ingress '$owner' on $APP_HOST; not overwriting" >&2
  exit 1
fi

echo "Deploying $APP_DIR as $APP_NAME to $NS at $APP_PATH"

code_args=(--from-file="$APP_DIR")
for f in "${EXTRA_FILES[@]}"; do code_args+=(--from-file="$f"); done
kubectl -n "$NS" create configmap "${APP_NAME}-code" "${code_args[@]}" \
  --dry-run=client -o yaml | kubectl apply -f -

creds() {
  printf 'VSS_URL=%s\nVSS_USERNAME=%s\nVSS_PASSWORD=%s\n' "$VSS_IN_CLUSTER" "$USERNAME" "$PASSWORD"
  for v in WANDB_API_KEY WANDB_TEAM WANDB_PROJECT; do
    [[ -n "${!v:-}" ]] && printf '%s=%s\n' "$v" "${!v}"
  done
  return 0
}
kubectl -n "$NS" create secret generic "${APP_NAME}-vss-creds" --from-env-file=<(creds) \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl -n "$NS" apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${APP_NAME}
  labels: {app: ${APP_NAME}}
spec:
  replicas: 1
  selector:
    matchLabels: {app: ${APP_NAME}}
  template:
    metadata:
      labels: {app: ${APP_NAME}}
    spec:
      containers:
      - name: app
        image: python:3.12-slim
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: ${APP_PORT}
        env:
        - {name: PORT, value: "${APP_PORT}"}
        - {name: BASE_PATH, value: "${APP_PATH}"}
        envFrom:
        - secretRef: {name: ${APP_NAME}-vss-creds}
        volumeMounts:
        - {name: code, mountPath: /code}
        workingDir: /code
        command: ["bash", "-c"]
        args:
        - |
          set -euo pipefail
          if [ -f requirements.txt ]; then pip install --no-cache-dir -q -r requirements.txt; fi
          exec python -u main.py
        readinessProbe:
          httpGet: {path: /health, port: ${APP_PORT}}
          initialDelaySeconds: 5
          periodSeconds: 10
      volumes:
      - name: code
        configMap: {name: ${APP_NAME}-code}
---
apiVersion: v1
kind: Service
metadata:
  name: ${APP_NAME}
  labels: {app: ${APP_NAME}}
spec:
  selector: {app: ${APP_NAME}}
  ports:
  - {name: http, port: 80, targetPort: ${APP_PORT}}
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ${APP_NAME}
  labels: {app: ${APP_NAME}}
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /\$2
spec:
  ingressClassName: nginx
  rules:
  - host: ${APP_HOST}
    http:
      paths:
      - path: ${APP_PATH}(/|\$)(.*)
        pathType: ImplementationSpecific
        backend:
          service:
            name: ${APP_NAME}
            port: {number: 80}
EOF

kubectl -n "$NS" rollout restart deploy/"$APP_NAME"
kubectl -n "$NS" rollout status deploy/"$APP_NAME" --timeout=180s

for url in "http://${PUBLIC_HOST}${APP_PATH}/" "http://${APP_HOST}${APP_PATH}/"; do
  code=000
  for _ in $(seq 1 12); do
    code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' "${url}health" || true)
    [[ "$code" == 200 ]] && break
    sleep 5
  done
  echo "$url -> health $code"
done
