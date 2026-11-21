{{- define "ev-battery.fullname" -}}
{{- printf "%s-ev-battery-api" .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "ev-battery.selectorLabels" -}}
app.kubernetes.io/name: ev-battery-api
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "ev-battery.labels" -}}
{{ include "ev-battery.selectorLabels" . }}
app.kubernetes.io/version: {{ .Values.image.tag | trunc 63 | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version }}
{{- end -}}
