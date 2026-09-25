# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""MCP-side ``submit_simulation`` orchestration (design sections 4.1, 8.2).

The service turns an LLM-supplied PICMI script into a ``Runner`` dump in a
disposable subprocess (the server never imports the script), wraps it in a
:class:`~pic_agentic.protocol.simulation.SimulationPayload`, embeds it in the
signed command and sends that.  It waits for the simclient's immediate
``accepted`` ack, so the LLM learns the ``sim_id`` right away; the later
lifecycle events (``simulation.submitted``/``results.ready``/
``simulation.failed``) are recorded as they arrive for the M2 reporting tools.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, computed_field

from pic_agentic.protocol.simulation import (
    SimulationPayload,
    SimulationType,
    SubmitParams,
    build_submit_command,
)
from pic_agentic.rcp import Kind, RcpMessage, SenderRole, SequenceState, new_cmd_id
from pic_agentic.server.hello import AckTimeoutError, SendFn
from pic_agentic.simulation_build import BuiltSimulation, SimulationBuildError, build_runner_dump

#: Signature of the injectable runner-dump builder (test seam).
RunnerDumpBuilder = Callable[..., Awaitable[BuiltSimulation]]


class SubmitOutcome(BaseModel):
    """The MCP-side result of one ``submit_simulation`` exchange."""

    model_config = ConfigDict(extra="forbid")

    sim: str
    cmd_id: str
    sim_id: str
    state: str
    job_id: int | None = None
    acked: bool = False
    error: str | None = None
    error_code: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ok(self) -> bool:
        """Whether the command was accepted without an error."""
        return not self.error


class SubmitService:
    """Turn PICMI scripts into commands and await their acks."""

    def __init__(
        self,
        sim: str,
        secret: str,
        *,
        picongpu_python: str = "",
        picongpu_revision: str = "",
        ack_timeout_s: float = 90.0,
        runner_dump_builder: RunnerDumpBuilder = build_runner_dump,
    ) -> None:
        """Create a service for one simulation.

        Args:
            sim: Simulation id.
            secret: Shared per-simulation RCP secret.
            picongpu_python: Interpreter with the pinned PIConGPU install.
            picongpu_revision: Pinned revision carried in the payload header.
            ack_timeout_s: Maximum wait for the ``accepted`` ack.
            runner_dump_builder: Subprocess runner-dump builder (test seam).

        """
        self.sim = sim
        self.secret = secret
        self.picongpu_python = picongpu_python
        self.picongpu_revision = picongpu_revision
        self.ack_timeout_s = ack_timeout_s
        self.runner_dump_builder = runner_dump_builder
        self.sequences = SequenceState()
        self._pending: dict[str, asyncio.Future[RcpMessage]] = {}
        #: Lifecycle events observed so far, keyed by cmd_id (M2 reporting).
        self.events: dict[str, list[RcpMessage]] = {}

    async def build_payload(
        self,
        script_path: Path,
        *,
        params: SubmitParams | None = None,
        cmd_id: str | None = None,
    ) -> tuple[str, SimulationPayload, RcpMessage]:
        """Build the payload and embed it in the signed command.

        Args:
            script_path: Path to the PICMI script (already resolved).
            params: Optional build/run flags.
            cmd_id: Optional command id (generated when omitted).

        Returns:
            The ``(cmd_id, payload, command)`` triple, the command signed.

        """
        command_id = cmd_id or new_cmd_id()
        built = await self.runner_dump_builder(script_path=script_path, interpreter=self.picongpu_python)
        # Provenance comes from the *child* that produced the dump, not from the
        # server process: the server may run a different interpreter (and, with
        # PIC_AGENTIC_PICONGPU_PYTHON, may not have PIConGPU at all).
        payload = SimulationPayload.build(
            picongpu_version=built.picongpu_version,
            picongpu_revision=self.picongpu_revision or built.picongpu_revision,
            schema_hash=built.schema_hash,
            runner_dump=built.runner,
        )
        payload.check_allowlist()
        seq = self.sequences.next_seq(self.sim, SenderRole.MCP_SERVER)
        command = build_submit_command(
            sim=self.sim,
            seq=seq,
            payload=payload,
            params=params,
            cmd_id=command_id,
        ).sign(self.secret)
        return command_id, payload, command

    def on_message(self, message: RcpMessage) -> None:
        """Feed an inbound message; resolve the pending ack and record events.

        Args:
            message: An inbound RCP message.

        """
        if not message.verify(self.secret) or message.sim != self.sim:
            return
        if message.sender_role is not SenderRole.SIMCLIENT:
            return
        cmd_id = str(message.payload.get("cmd_id", ""))
        if message.kind is Kind.EVENT and message.type == SimulationType.EVENT:
            self.events.setdefault(cmd_id, []).append(message)
            return
        if message.kind is Kind.ACK and message.type == SimulationType.ACK:
            future = self._pending.get(cmd_id)
            if future is not None and not future.done():
                future.set_result(message)

    def _outcome_from_ack(self, cmd_id: str, ack: RcpMessage) -> SubmitOutcome:
        return SubmitOutcome(
            sim=self.sim,
            cmd_id=cmd_id,
            sim_id=str(ack.payload.get("sim_id", "")),
            state=str(ack.payload.get("state", "")),
            job_id=ack.payload.get("job_id"),
            acked=True,
            error=ack.payload.get("error"),
            error_code=ack.payload.get("error_code"),
        )

    async def submit(
        self,
        send: SendFn,
        script_path: Path,
        *,
        params: SubmitParams | None = None,
    ) -> SubmitOutcome:
        """Build, send and await one ``submit_simulation`` command.

        Args:
            send: Async callable ``send(RcpMessage) -> event_id``.
            script_path: Path to the PICMI script.
            params: Optional build/run flags.

        Returns:
            The outcome; ``state`` is the simclient's first ack state (normally
            ``accepted``).

        Raises:
            AckTimeoutError: If no ack arrives within the configured wait.

        """
        cmd_id, _payload, command = await self.build_payload(script_path, params=params)
        future: asyncio.Future[RcpMessage] = asyncio.get_running_loop().create_future()
        self._pending[cmd_id] = future
        try:
            await send(command)
            try:
                ack = await asyncio.wait_for(future, timeout=self.ack_timeout_s)
            except TimeoutError:
                msg = f"no ack for submit command {cmd_id} within {self.ack_timeout_s}s"
                raise AckTimeoutError(msg) from None
        finally:
            self._pending.pop(cmd_id, None)
        return self._outcome_from_ack(cmd_id, ack)


def resolve_script(picmi_script: str, *, workdir: Path) -> Path:
    """Resolve the tool's ``picmi_script`` argument to a file path.

    Args:
        picmi_script: An existing file path, or inline PICMI code.
        workdir: Directory for inline code (a temp dir on the server).

    Returns:
        The path to the PICMI script.

    """
    candidate = Path(picmi_script).expanduser()
    if "\n" not in picmi_script and candidate.is_file():
        return candidate
    workdir.mkdir(parents=True, exist_ok=True)
    # A unique name so concurrent submissions never overwrite each other.
    fd, name = tempfile.mkstemp(prefix="picmi_script-", suffix=".py", dir=str(workdir))
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(picmi_script)
    return Path(name)


__all__ = [
    "AckTimeoutError",
    "SimulationBuildError",
    "SubmitOutcome",
    "SubmitService",
    "resolve_script",
]
