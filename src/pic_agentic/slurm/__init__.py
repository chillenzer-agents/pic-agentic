"""SLURM command layer (design section 2.4).

Only the fixed command set is exposed; no arbitrary shell tool.  Every value
that reaches a shell is either a server-generated identifier with a safe
charset or an absolute path to a server-controlled file (design section 6.4).
"""

from pic_agentic.slurm.client import SlurmClient, SlurmError, SlurmJobState

__all__ = ["SlurmClient", "SlurmError", "SlurmJobState"]
