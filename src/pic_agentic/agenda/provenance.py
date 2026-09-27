# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Milestone E: campaign-level provenance as a minimal RO-Crate document.

A campaign is a research act: inputs (the specs), runs (the submissions) and
analyses (the findings) linked at a pinned revision.  This module renders that
lineage as an RO-Crate 1.1 ``ro-crate-metadata.json`` JSON-LD document -- the
same format :func:`pic_agentic.analysis.read_rocrate` reads -- so the record can
be attached to a report or archived without a bespoke schema.

The function is a pure function of the persisted campaign (plus an optional
analysis mapping), extraction-ready (stdlib + the agenda model only).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pic_agentic.agenda.campaign import Campaign

#: The RO-Crate metadata descriptor file name (the crate's entry point).
RO_CRATE_METADATA_FILE = "ro-crate-metadata.json"

#: RO-Crate context URI.
_RO_CRATE_CONTEXT = "https://w3id.org/ro/crate/1.1/context"

#: The crate's conformance URI for the metadata descriptor.
_RO_CRATE_PROFILE = "https://w3id.org/ro/crate/1.1"


def sanitize_id(path: str) -> str:
    """Return a fragment-safe, collision-free ``@id`` for a leaf path.

    ``/`` is percent-encoded (along with ``%`` itself), so the map is injective:
    distinct leaf paths always yield distinct ids (unlike a plain ``/`` -> ``_``
    substitution, under which ``a/b`` and ``a_b`` would collide).

    Args:
        path: The agenda leaf path (``a/b/c``).

    Returns:
        A ``#``-prefixed fragment with separators percent-encoded.

    """
    safe = path.replace("%", "%25").replace("/", "%2F")
    return f"#{safe}"


def campaign_rocrate(
    campaign: Campaign,
    *,
    analyses: Mapping[str, Mapping[str, Any]] | None = None,
    revision: str | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """Render a campaign's lineage as a minimal RO-Crate JSON-LD document.

    Args:
        campaign: The persisted campaign.
        analyses: Optional ``leaf path -> analysis dict`` mapping (e.g. the
            ``analyze_output`` sections); each linked as a derived entity.
        revision: The pinned PIConGPU revision; when omitted, a uniform
            spec-carried revision is used.
        name: Optional crate name (defaults to the campaign name).

    Returns:
        The ``ro-crate-metadata.json`` document as a dict.

    """
    analyses = analyses or {}
    pinned = revision or _uniform_revision(campaign)
    crate_name = name or campaign.name
    software_id = "#picongpu"
    graph: list[dict[str, Any]] = [
        {
            "@id": RO_CRATE_METADATA_FILE,
            "@type": "CreativeWork",
            "about": {"@id": "./"},
            "conformsTo": {"@id": _RO_CRATE_PROFILE},
        },
        {
            "@id": "./",
            "@type": "Dataset",
            "name": crate_name,
            "datePublished": campaign.created_ts,
            "hasPart": [{"@id": sanitize_id(path)} for path, _ in campaign.agenda.simulations()],
            "mentions": {"@id": software_id},
        },
        {
            "@id": software_id,
            "@type": "SoftwareSourceCode",
            "name": "PIConGPU",
            "version": pinned or "",
            "identifier": pinned or "",
        },
    ]
    for path, sim in campaign.agenda.simulations():
        leaf_id = sanitize_id(path)
        entities: list[dict[str, Any]] = [
            {
                "@id": leaf_id,
                "@type": "File",
                "name": path,
                "sha256": _spec_hash(sim.spec),
                "identifier": sim.sim_id or path,
                "description": f"status={sim.status}",
                "about": {"@id": software_id},
                "instrument": {"@id": software_id},
            },
            {
                "@id": f"{leaf_id}/run",
                "@type": "CreateAction",
                "name": f"run {path}",
                "object": {"@id": leaf_id},
                "instrument": {"@id": software_id},
            },
        ]
        analysis = analyses.get(path)
        if analysis is not None:
            analysis_id = f"{leaf_id}/analysis"
            entities.extend(
                (
                    {
                        "@id": analysis_id,
                        "@type": "Dataset",
                        "name": f"analysis {path}",
                        "sha256": _json_hash(analysis),
                    },
                    {
                        "@id": f"{leaf_id}/analysis_action",
                        "@type": "CreateAction",
                        "name": f"analyse {path}",
                        "object": {"@id": leaf_id},
                        "result": {"@id": analysis_id},
                    },
                ),
            )
        graph.extend(entities)
    return {"@context": _RO_CRATE_CONTEXT, "@graph": graph}


def _uniform_revision(campaign: Campaign) -> str:
    """Return the single spec-carried revision, or an empty string.

    Returns:
        The shared ``picongpu_revision`` when every leaf that carries one agrees,
        else an empty string.

    """
    revisions = {
        str(sim.spec["provenance"]["picongpu_revision"])
        for _, sim in campaign.agenda.simulations()
        if isinstance(sim.spec.get("provenance"), Mapping) and sim.spec["provenance"].get("picongpu_revision")
    }
    return next(iter(revisions)) if len(revisions) == 1 else ""


def _spec_hash(spec: Mapping[str, Any]) -> str:
    """Return the stable content hash of a leaf spec.

    Returns:
        The sha256 hex digest of the canonical JSON encoding.

    """
    return _json_hash(spec)


def _json_hash(value: Any) -> str:
    """Return a stable sha256 over canonical JSON.

    Returns:
        The hex digest.

    """
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ``leaf_at`` is available from :mod:`pic_agentic.agenda.engine` for callers that
# resolve paths against an agenda.
__all__ = ["RO_CRATE_METADATA_FILE", "campaign_rocrate", "sanitize_id"]
