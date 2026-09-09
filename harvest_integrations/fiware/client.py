"""
Minimal NGSI-LD context-broker client.

Standard library only (``urllib``), matching HARVEST's no-framework server
philosophy: the broker API surface we need is small (batch upsert, get, patch,
subscriptions), and keeping ``requests`` out of the core dependency set means
the FIWARE layer adds *zero* mandatory third-party packages.

Tested against Orion-LD (the broker shipped in docker-compose.yml); the calls
are plain NGSI-LD 1.x, so other brokers (Scorpio, Stellio) should work
unchanged.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

CORE_CONTEXT = "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context-v1.8.jsonld"


class ContextBrokerError(RuntimeError):
    def __init__(self, message: str, status: Optional[int] = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


class ContextBrokerClient:
    def __init__(self, base_url: str, timeout_s: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    # -- low-level ------------------------------------------------------------
    def _request(
        self,
        method: str,
        path: str,
        payload: Any = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        data = None
        hdrs = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            hdrs["Content-Type"] = "application/json"
            # Terms expand under the NGSI-LD default vocabulary; the Link
            # header keeps payloads in plain application/json.
            hdrs["Link"] = (
                f'<{CORE_CONTEXT}>; '
                'rel="http://www.w3.org/ns/json-ld#context"; '
                'type="application/ld+json"'
            )
        if headers:
            hdrs.update(headers)
        req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = resp.read().decode("utf-8", "replace")
                return json.loads(body) if body.strip() else None
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            raise ContextBrokerError(
                f"{method} {path} -> HTTP {exc.code}: {body[:300]}",
                status=exc.code, body=body,
            ) from exc
        except urllib.error.URLError as exc:
            raise ContextBrokerError(f"{method} {path} -> {exc.reason}") from exc

    # -- broker surface -------------------------------------------------------
    def version(self) -> Dict[str, Any]:
        return self._request("GET", "/version") or {}

    def is_alive(self) -> bool:
        try:
            self.version()
            return True
        except ContextBrokerError:
            return False

    def upsert_entities(self, entities: List[Dict[str, Any]]) -> None:
        """Batch create-or-update (attribute values replaced)."""
        if not entities:
            return
        self._request(
            "POST", "/ngsi-ld/v1/entityOperations/upsert?options=update", entities)

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        try:
            return self._request(
                "GET", f"/ngsi-ld/v1/entities/{entity_id}?options=keyValues")
        except ContextBrokerError as exc:
            if exc.status == 404:
                return None
            raise

    def list_entities(self, entity_type: str) -> List[Dict[str, Any]]:
        return self._request(
            "GET", f"/ngsi-ld/v1/entities?type={entity_type}&options=keyValues") or []

    def patch_attrs(self, entity_id: str, attrs: Dict[str, Any]) -> None:
        self._request("PATCH", f"/ngsi-ld/v1/entities/{entity_id}/attrs", attrs)

    # -- subscriptions --------------------------------------------------------
    def ensure_subscription(self, subscription: Dict[str, Any]) -> None:
        """Create the subscription, replacing any previous one with this id."""
        sub_id = subscription["id"]
        try:
            self._request("DELETE", f"/ngsi-ld/v1/subscriptions/{sub_id}")
        except ContextBrokerError as exc:
            if exc.status not in (404, None):
                raise
        self._request("POST", "/ngsi-ld/v1/subscriptions", subscription)

    def delete_subscription(self, sub_id: str) -> None:
        try:
            self._request("DELETE", f"/ngsi-ld/v1/subscriptions/{sub_id}")
        except ContextBrokerError as exc:
            if exc.status != 404:
                raise
