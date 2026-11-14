from datetime import UTC, datetime

from pydantic import Field

from agent.schemas import StrictInput


class NamespaceInput(StrictInput):
    namespace: str = "mlops"


class PodInput(NamespaceInput):
    pod: str = Field(pattern=r"^[a-z0-9][a-z0-9.-]{0,252}$")
    tail_lines: int = Field(default=80, ge=1, le=200)


class RestartInput(NamespaceInput):
    deployment: str = "prediction-api"


class KubernetesTools:
    def __init__(self, core, apps, namespace: str, deployment: str, timeout: float = 20):
        self.core, self.apps = core, apps
        self.namespace, self.deployment, self.timeout = namespace, deployment, timeout

    def _scope(self, namespace: str):
        if namespace != self.namespace:
            raise PermissionError("Namespace is outside the allowlist")

    def pods(self, namespace: str = "mlops") -> dict:
        self._scope(namespace)
        result = self.core.list_namespaced_pod(
            namespace, label_selector="app=prediction-api", _request_timeout=self.timeout
        )
        return {
            "namespace": namespace,
            "pods": [
                {
                    "name": p.metadata.name,
                    "phase": p.status.phase,
                    "conditions": [
                        {"type": c.type, "status": c.status, "reason": c.reason}
                        for c in (p.status.conditions or [])
                    ],
                    "containers": [
                        {
                            "name": c.name,
                            "ready": c.ready,
                            "restarts": c.restart_count,
                            "waiting_reason": c.state.waiting.reason if c.state.waiting else None,
                            "last_exit_code": c.last_state.terminated.exit_code
                            if c.last_state and c.last_state.terminated
                            else None,
                        }
                        for c in (p.status.container_statuses or [])
                    ],
                }
                for p in result.items
            ],
        }

    def deployments(self, namespace: str = "mlops") -> dict:
        self._scope(namespace)
        result = self.apps.list_namespaced_deployment(
            namespace, label_selector="app=prediction-api", _request_timeout=self.timeout
        )
        return {
            "namespace": namespace,
            "deployments": [
                {
                    "name": d.metadata.name,
                    "desired": d.spec.replicas,
                    "ready": d.status.ready_replicas or 0,
                    "available": d.status.available_replicas or 0,
                    "generation": d.metadata.generation,
                    "observed_generation": d.status.observed_generation,
                    "images": [c.image for c in d.spec.template.spec.containers],
                    "model_run_id": (d.spec.template.metadata.annotations or {}).get(
                        "mlops/model-run-id"
                    ),
                    "conditions": [
                        {"type": c.type, "status": c.status, "reason": c.reason}
                        for c in (d.status.conditions or [])
                    ],
                }
                for d in result.items
            ],
        }

    def services(self, namespace: str = "mlops") -> dict:
        self._scope(namespace)
        result = self.core.list_namespaced_service(
            namespace, label_selector="app=prediction-api", _request_timeout=self.timeout
        )
        return {
            "namespace": namespace,
            "services": [
                {
                    "name": s.metadata.name,
                    "type": s.spec.type,
                    "cluster_ip": s.spec.cluster_ip,
                    "ports": [{"port": p.port, "target_port": p.target_port} for p in s.spec.ports],
                }
                for s in result.items
            ],
        }

    def events(self, namespace: str = "mlops") -> dict:
        self._scope(namespace)
        events = self.core.list_namespaced_event(namespace, _request_timeout=self.timeout)
        # Only application events; never dump other workloads in the namespace.
        relevant = [e for e in events.items if e.involved_object.name.startswith(self.deployment)]
        return {
            "events": [
                {
                    "object": e.involved_object.name,
                    "reason": e.reason,
                    "message": e.message,
                    "count": e.count,
                    "type": e.type,
                }
                for e in relevant[-30:]
            ]
        }

    def pod_logs(self, pod: str, namespace: str = "mlops", tail_lines: int = 80) -> dict:
        self._scope(namespace)
        item = self.core.read_namespaced_pod(pod, namespace, _request_timeout=self.timeout)
        if (item.metadata.labels or {}).get("app") != "prediction-api":
            raise PermissionError("Pod is outside the application allowlist")
        return {
            "pod": pod,
            "logs": self.core.read_namespaced_pod_log(
                pod,
                namespace,
                tail_lines=tail_lines,
                limit_bytes=16000,
                _request_timeout=self.timeout,
            ),
        }

    def restart_deployment(
        self, deployment: str = "prediction-api", namespace: str = "mlops"
    ) -> dict:
        self._scope(namespace)
        if deployment != self.deployment:
            raise PermissionError("Deployment is outside the allowlist")
        self.apps.patch_namespaced_deployment(
            deployment,
            namespace,
            {
                "spec": {
                    "template": {
                        "metadata": {
                            "annotations": {
                                "kubectl.kubernetes.io/restartedAt": datetime.now(UTC).isoformat()
                            }
                        }
                    }
                }
            },
            _request_timeout=self.timeout,
        )
        return {
            "deployment": deployment,
            "namespace": namespace,
            "restart_requested": True,
            "note": "Observe rollout; this does not prove recovery.",
        }
