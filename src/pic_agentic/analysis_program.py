# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""A safe, declarative analysis program (no code execution).

This is the wire format for "tailored analysis on the cluster" without arbitrary
code execution.  It is deliberately **not** a serialised ``sympy`` object or
string: ``srepr`` is meant to be ``eval``'d, ``pickle`` executes on load, and
``sympify``/``parse_expr`` are not security boundaries.  Instead a program is a
small, fully validated **expression tree** of a fixed node set, evaluated by
:mod:`pic_agentic.analysis_eval` with stdlib math.

The model is fail-closed: every node forbids unknown fields, the node/function
vocabulary is closed, and a whole-program validator enforces hard limits
(node count, depth, selector count, exponent, bins, output size).  A program
that fails validation never reaches the evaluator.

Extraction-ready: stdlib + pydantic only.
"""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: Hard caps enforced on every program.
MAX_NODES = 256
MAX_DEPTH = 32
MAX_SELECTORS = 8
MAX_EXPONENT = 12
MAX_BINS = 4096
MAX_POINTS_OUT = 4096
#: Cap on the serialised program itself, so a giant literal list cannot be sent.
MAX_SOURCE_BYTES = 48 * 1024

BinaryOp = Literal["add", "sub", "mul", "div", "pow"]
UnaryOp = Literal["neg", "abs", "sqrt", "log", "exp", "sin", "cos", "tanh", "sign"]
ReduceOp = Literal[
    "sum",
    "mean",
    "std",
    "min",
    "max",
    "median",
    "argmax",
    "argmin",
    "quantile",
    "histogram",
    "fft_peak",
    "fft_freq",
]


class _Node(BaseModel):
    """Base for every AST node: no unknown fields, so the vocabulary is closed."""

    model_config = ConfigDict(extra="forbid")


class Const(_Node):
    """A numeric literal."""

    kind: Literal["const"] = "const"
    value: float

    @field_validator("value")
    @classmethod
    def _finite(cls, value: float) -> float:
        """Reject a non-finite literal (it would poison the evaluation).

        Returns:
            The validated value.

        Raises:
            ValueError: If the value is not finite.

        """
        if not math.isfinite(value):
            msg = f"constant must be finite, got {value!r}"
            raise ValueError(msg)
        return value


class VarRef(_Node):
    """A named selector bound to one openPMD mesh component.

    The selector is *data*, not a path or command: the cluster resolves it with
    the same allow-listed record/component/iteration logic as the existing
    result reads, so it cannot name an arbitrary file.
    """

    kind: Literal["var"] = "var"
    name: str
    record: str | None = None
    component: str | None = None
    iteration: int | str | None = None

    @field_validator("name", "record", "component")
    @classmethod
    def _name_safe(cls, value: str | None) -> str | None:
        """Restrict selector names to a conservative identifier charset.

        Returns:
            The validated name.

        Raises:
            ValueError: If the name carries an unsafe character.

        """
        if value is None:
            return None
        if not value or not all(char.isalnum() or char in "._-" for char in value):
            msg = f"unsafe selector name: {value!r}"
            raise ValueError(msg)
        return value


class BinOp(_Node):
    """A binary arithmetic operation."""

    kind: Literal["binop"] = "binop"
    op: BinaryOp
    left: Expr
    right: Expr


class UnOp(_Node):
    """A unary function of one operand."""

    kind: Literal["unop"] = "unop"
    op: UnaryOp
    operand: Expr


class Reduce(_Node):
    """A reduction of an operand (usually an array-valued selector)."""

    kind: Literal["reduce"] = "reduce"
    op: ReduceOp
    operand: Expr
    #: Quantile in [0, 1] (``quantile`` only).
    q: float | None = None
    #: Bin count (``histogram`` only).
    bins: int | None = None

    @field_validator("q")
    @classmethod
    def _q_range(cls, value: float | None) -> float | None:
        """Keep a quantile inside [0, 1].

        Returns:
            The validated quantile, or None.

        Raises:
            ValueError: If the quantile is outside [0, 1].

        """
        if value is None:
            return None
        if not 0.0 <= value <= 1.0:
            msg = f"quantile q must be in [0, 1], got {value!r}"
            raise ValueError(msg)
        return value

    @field_validator("bins")
    @classmethod
    def _bins_range(cls, value: int | None) -> int | None:
        """Keep a histogram bin count positive and bounded.

        Returns:
            The validated bin count, or None.

        Raises:
            ValueError: If the bin count is out of range.

        """
        if value is None:
            return None
        if not 1 <= value <= MAX_BINS:
            msg = f"bins must be in [1, {MAX_BINS}], got {value!r}"
            raise ValueError(msg)
        return value


#: The closed expression vocabulary, discriminated on ``kind``.
Expr = Annotated[Const | VarRef | BinOp | UnOp | Reduce, Field(discriminator="kind")]

#: A node's child expressions, in evaluation order.
_CHILDREN: dict[str, tuple[str, ...]] = {
    "const": (),
    "var": (),
    "binop": ("left", "right"),
    "unop": ("operand",),
    "reduce": ("operand",),
}


class AnalysisProgram(BaseModel):
    """A complete, validated analysis program.

    ``output`` is the value returned (a scalar or, if it references array-valued
    selectors, an array); an optional ``selectors`` list documents the named
    inputs.  Reductions turn an array into a scalar, so the common case is a
    program whose ``output`` is a ``reduce`` node over a ``var`` selector.

    A ``var`` node may carry ``record``/``component``/``iteration`` directly;
    those attributes are authoritative (a missing one falls back to the matching
    declared selector, then to name-only resolution).  A node attribute that
    *contradicts* the declared selector is rejected at validation time rather
    than silently preferring one, because a conflict means the program says two
    different things about the same input.
    """

    model_config = ConfigDict(extra="forbid")

    output: Expr
    selectors: list[VarRef] = Field(default_factory=list)
    #: Optional second expression evaluated in parallel with ``output`` (e.g. an
    #: FFT frequency axis alongside its peak magnitudes).
    points: Expr | None = None

    @field_validator("selectors")
    @classmethod
    def _selector_cap(cls, value: list[VarRef]) -> list[VarRef]:
        """Bound the number of declared selectors.

        Returns:
            The validated selector list.

        Raises:
            ValueError: If too many selectors are declared.

        """
        if len(value) > MAX_SELECTORS:
            msg = f"at most {MAX_SELECTORS} selectors are allowed, got {len(value)}"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _enforce_limits(self) -> AnalysisProgram:
        """Enforce the structural limits over the whole tree.

        Returns:
            The validated program.

        Raises:
            ValueError: If any limit is exceeded or an exponent is too large.

        """
        nodes = 0
        for expression in self._expressions():
            count, depth, exponent = _measure(expression)
            nodes += count
            if nodes > MAX_NODES:
                msg = f"program exceeds {MAX_NODES} nodes"
                raise ValueError(msg)
            if depth > MAX_DEPTH:
                msg = f"program depth {depth} exceeds {MAX_DEPTH}"
                raise ValueError(msg)
            if exponent > MAX_EXPONENT:
                msg = f"exponent {exponent} exceeds {MAX_EXPONENT}"
                raise ValueError(msg)
        self._reject_attribute_conflicts()
        return self

    def _reject_attribute_conflicts(self) -> None:
        """Reject a ``var`` attribute that contradicts its declared selector.

        Node-level ``record``/``component``/``iteration`` are authoritative, but
        a node that names a declared selector with a *different* value for the
        same attribute is ambiguous -- the reader would use the node value while
        the declaration says otherwise.

        Raises:
            ValueError: If any referenced ``var`` conflicts with its declaration.

        """
        declared = {selector.name: selector for selector in self.selectors}
        for expression in self._expressions():
            for ref in _iter_var_refs(expression):
                selector = declared.get(ref.name)
                if selector is None:
                    continue
                for field in ("record", "component", "iteration"):
                    node_value = getattr(ref, field)
                    declared_value = getattr(selector, field)
                    if node_value is not None and declared_value is not None and node_value != declared_value:
                        msg = (
                            f"var {ref.name!r} sets {field}={node_value!r} but its declared selector "
                            f"sets {field}={declared_value!r}"
                        )
                        raise ValueError(msg)

    def _expressions(self) -> list[Any]:
        """Return the root expressions of this program.

        Returns:
            ``[output]`` plus ``points`` when set.

        """
        return [self.output, self.points] if self.points is not None else [self.output]


def _iter_var_refs(node: Any) -> list[VarRef]:
    """Collect every ``var`` node under ``node``, in traversal order.

    Returns:
        The referenced ``VarRef`` nodes.

    """
    refs: list[VarRef] = []
    stack = [node]
    while stack:
        current = stack.pop()
        if getattr(current, "kind", None) == "var":
            refs.append(current)
        stack.extend(getattr(current, field) for field in _CHILDREN.get(getattr(current, "kind", None), ()))
    return refs


def _measure(node: Any, depth: int = 1) -> tuple[int, int, int]:
    """Measure a subtree: node count, max depth and max exponent magnitude.

    Walks the validated node objects generically (using the child-field table),
    so a new node type is measured without touching this function.

    Args:
        node: A validated AST node.
        depth: The current depth (root is 1).

    Returns:
        A ``(nodes, depth, exponent)`` triple.

    """
    kind = getattr(node, "kind", None)
    count = 1
    max_depth = depth
    exponent = _node_exponent(node, kind)
    for field in _CHILDREN.get(kind, ()):
        child = getattr(node, field)
        child_nodes, child_depth, child_exponent = _measure(child, depth + 1)
        count += child_nodes
        max_depth = max(max_depth, child_depth)
        exponent = max(exponent, child_exponent)
    return count, max_depth, exponent


def _node_exponent(node: Any, kind: str | None) -> int:
    """Return the exponent magnitude a node contributes, if any.

    A ``pow`` with a literal constant exponent is bounded here; a non-literal
    exponent is bounded at evaluation time (the evaluator refuses to raise a
    value to a non-constant power, which would be unbounded over an array).

    Returns:
        The exponent magnitude, or 0 when the node sets none.

    """
    if kind == "binop" and getattr(node, "op", None) == "pow" and isinstance(node.right, Const):
        return int(abs(node.right.value))
    return 0


# Resolve the recursive forward references now that the union is defined.
BinOp.model_rebuild()
UnOp.model_rebuild()
Reduce.model_rebuild()
AnalysisProgram.model_rebuild()


__all__ = [
    "MAX_BINS",
    "MAX_DEPTH",
    "MAX_EXPONENT",
    "MAX_NODES",
    "MAX_POINTS_OUT",
    "MAX_SELECTORS",
    "MAX_SOURCE_BYTES",
    "AnalysisProgram",
    "BinOp",
    "BinaryOp",
    "Const",
    "Expr",
    "Reduce",
    "ReduceOp",
    "UnOp",
    "UnaryOp",
    "VarRef",
]
