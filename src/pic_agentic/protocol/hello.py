"""M1 ``hello`` command/ack RCP messages (design sections 2.4, 8.1).

Shell-safety contract (design section 6.4): the LLM-supplied ``message`` never
reaches a shell.  The MCP server picks a server-generated absolute path, the
simclient writes the message there, and the job body is ``cat '<path>'``.
"""

from __future__ import annotations

from pic_agentic.rcp import Kind, RcpMessage, SenderRole, new_cmd_id

HELLO = "rcp.hello"
HELLO_ACK = "rcp.hello_ack"

DEFAULT_MESSAGE = "Hello World"


def build_hello_command(
    *,
    sim: str,
    seq: int,
    message_path: str,
    message: str = DEFAULT_MESSAGE,
    cmd_id: str | None = None,
    in_reply_to: str | None = None,
) -> RcpMessage:
    """MCP server -> simclient command.

    The payload carries the LLM-supplied ``message`` as *data* and a
    server-generated absolute ``message_path``.  The simclient writes the
    message to that path; the message text is never passed to a shell.
    """
    return RcpMessage(
        sim=sim,
        kind=Kind.COMMAND,
        type=HELLO,
        seq=seq,
        sender_role=SenderRole.MCP_SERVER,
        in_reply_to=in_reply_to,
        payload={"cmd_id": cmd_id or new_cmd_id(), "message": message, "message_path": message_path},
    )


def build_hello_ack(
    *,
    sim: str,
    seq: int,
    cmd_id: str,
    in_reply_to: str | None,
    job_id: int | None,
    cluster_output: str | None,
    error: str | None = None,
) -> RcpMessage:
    """simclient -> room acknowledgement of a ``hello`` command."""
    payload: dict[str, object] = {
        "cmd_id": cmd_id,
        "job_id": job_id,
        "cluster_output": cluster_output,
    }
    if error:
        payload["error"] = error
    return RcpMessage(
        sim=sim,
        kind=Kind.ACK,
        type=HELLO_ACK,
        seq=seq,
        sender_role=SenderRole.SIMCLIENT,
        in_reply_to=in_reply_to,
        payload=payload,
    )
