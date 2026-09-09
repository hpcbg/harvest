"""
harvest_integrations.fiware
===========================

FIWARE / NGSI-LD northbound integration.

HARVEST's semantic device-agent model (``harvest_control``) stays the
authoritative domain model; the context broker (Orion-LD) holds a *mirror* of
current farm state plus one inbound command entity, following the WISEPACK
state-oriented broker pattern (the broker is context, not a log).

* :mod:`client`   -- minimal NGSI-LD HTTP client (standard library only);
* :mod:`entities` -- mapping between fleet snapshots and NGSI-LD entities;
* :mod:`sync`     -- the synchronisation daemon
  (``python -m harvest_integrations.fiware.sync``).
"""
from .client import ContextBrokerClient, ContextBrokerError
from .entities import (
    COMMAND_ENTITY_ID,
    command_entity,
    entity_id,
    snapshot_to_entities,
)

__all__ = [
    "ContextBrokerClient",
    "ContextBrokerError",
    "COMMAND_ENTITY_ID",
    "command_entity",
    "entity_id",
    "snapshot_to_entities",
]
