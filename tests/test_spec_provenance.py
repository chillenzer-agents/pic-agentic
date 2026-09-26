# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Tests for the direct-spec submission provenance fallback (C-full M6)."""

from __future__ import annotations

from pic_agentic.server.simulation import _spec_provenance


def test_local_provenance_preferred(monkeypatch) -> None:
    monkeypatch.setattr(
        "pic_agentic.server.simulation.local_provenance",
        lambda: {"picongpu_version": "0.9", "picongpu_revision": "local", "schema_hash": "abc"},
    )
    spec = {"sim": {}, "provenance": {"picongpu_revision": "carried"}}
    out = _spec_provenance(spec, "")
    assert out["picongpu_revision"] == "local"


def test_spec_carried_provenance_is_the_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        "pic_agentic.server.simulation.local_provenance",
        lambda: {"picongpu_version": "", "picongpu_revision": "", "schema_hash": ""},
    )
    spec = {
        "sim": {},
        "provenance": {"picongpu_version": "0.9", "picongpu_revision": "carried", "schema_hash": "deadbeef"},
    }
    out = _spec_provenance(spec, "")
    assert out == {"picongpu_version": "0.9", "picongpu_revision": "carried", "schema_hash": "deadbeef"}


def test_configured_revision_wins_over_carried(monkeypatch) -> None:
    monkeypatch.setattr(
        "pic_agentic.server.simulation.local_provenance",
        lambda: {"picongpu_version": "", "picongpu_revision": "", "schema_hash": ""},
    )
    spec = {"sim": {}, "provenance": {"picongpu_revision": "carried"}}
    out = _spec_provenance(spec, "configured")
    assert out["picongpu_revision"] == "configured"


def test_no_provenance_anywhere_is_empty(monkeypatch) -> None:
    monkeypatch.setattr(
        "pic_agentic.server.simulation.local_provenance",
        lambda: {"picongpu_version": "", "picongpu_revision": "", "schema_hash": ""},
    )
    assert _spec_provenance({"sim": {}}, "") == {
        "picongpu_version": "",
        "picongpu_revision": "",
        "schema_hash": "",
    }
