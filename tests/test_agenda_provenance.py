# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Tests for the campaign RO-Crate provenance (milestone E)."""

from __future__ import annotations

from pic_agentic.agenda.campaign import Campaign
from pic_agentic.agenda.model import AgendaGroup, AgendaSim
from pic_agentic.agenda.provenance import RO_CRATE_METADATA_FILE, campaign_rocrate, sanitize_id


def _campaign(*specs: dict, name: str = "study") -> Campaign:
    group = AgendaGroup(name="group")
    for index, spec in enumerate(specs):
        leaf = f"leaf{index}"
        group = group.add(**{leaf: AgendaSim(name=leaf, spec=spec)})
    return Campaign(name=name, agenda=group).with_created_ts()


def _by_id(crate: dict, entity_id: str) -> dict:
    return next(entity for entity in crate["@graph"] if entity["@id"] == entity_id)


def test_required_entities_present() -> None:
    crate = campaign_rocrate(_campaign({"sim": {"replica": 0}}))
    assert crate["@context"].startswith("https://w3id.org/ro/crate/1.1/context")
    descriptor = _by_id(crate, RO_CRATE_METADATA_FILE)
    assert descriptor["about"] == {"@id": "./"}
    root = _by_id(crate, "./")
    assert root["@type"] == "Dataset"
    assert root["name"] == "study"
    assert root["datePublished"]
    assert root["mentions"] == {"@id": "#picongpu"}
    software = _by_id(crate, "#picongpu")
    assert software["@type"] == "SoftwareSourceCode"
    assert software["name"] == "PIConGPU"


def test_one_file_per_leaf_with_spec_hash() -> None:
    crate = campaign_rocrate(_campaign({"sim": {"replica": 0}}, {"sim": {"replica": 1}}))
    root = _by_id(crate, "./")
    assert root["hasPart"] == [{"@id": "#leaf0"}, {"@id": "#leaf1"}]
    leaf = _by_id(crate, "#leaf0")
    assert leaf["@type"] == "File"
    assert leaf["name"] == "leaf0"
    assert len(leaf["sha256"]) == 64
    # The leaf's spec hash matches a direct computation of the same canonical JSON.
    from pic_agentic.agenda.provenance import _spec_hash

    assert leaf["sha256"] == _spec_hash({"sim": {"replica": 0}})


def test_run_action_links_leaf_to_software() -> None:
    crate = campaign_rocrate(_campaign({"sim": {"replica": 0}}))
    run = _by_id(crate, "#leaf0/run")
    assert run["@type"] == "CreateAction"
    assert run["object"] == {"@id": "#leaf0"}
    assert run["instrument"] == {"@id": "#picongpu"}


def test_analysis_linkage() -> None:
    crate = campaign_rocrate(
        _campaign({"sim": {"replica": 0}}),
        analyses={"leaf0": {"answer": "peak at 1e19"}},
    )
    analysis = _by_id(crate, "#leaf0/analysis")
    assert analysis["@type"] == "Dataset"
    assert len(analysis["sha256"]) == 64
    action = _by_id(crate, "#leaf0/analysis_action")
    assert action["result"] == {"@id": "#leaf0/analysis"}
    assert action["object"] == {"@id": "#leaf0"}


def test_revision_override_and_spec_carried() -> None:
    crate = campaign_rocrate(_campaign({"sim": {"replica": 0}}), revision="deadbeef")
    assert _by_id(crate, "#picongpu")["version"] == "deadbeef"

    carried = _campaign({"sim": {"replica": 0}, "provenance": {"picongpu_revision": "abc123"}})
    assert _by_id(campaign_rocrate(carried), "#picongpu")["identifier"] == "abc123"


def test_mixed_revisions_leave_the_pin_blank() -> None:
    mixed = _campaign(
        {"sim": {"replica": 0}, "provenance": {"picongpu_revision": "a"}},
        {"sim": {"replica": 1}, "provenance": {"picongpu_revision": "b"}},
    )
    assert not _by_id(campaign_rocrate(mixed), "#picongpu")["version"]


def test_sanitize_id_replaces_separators() -> None:
    assert sanitize_id("group/leaf") == "#group_leaf"
    assert sanitize_id("leaf") == "#leaf"


def test_crate_is_json_serialisable() -> None:
    import json

    crate = campaign_rocrate(_campaign({"sim": {"replica": 0}}), analyses={"leaf0": {"answer": "x"}})
    assert json.loads(json.dumps(crate)) == crate
