"""Portable filter language (FR-S2).

User syntax (one operator per dict)::

    {"eq": {"tag_product": "billing"}}
    {"ne": {"quarantined": True}}
    {"in": {"tag_doc_type": ["policy", "faq"]}}
    {"nin": {...}}
    {"gt" | "gte" | "lt" | "lte": {"chunk_index": 3}}
    {"exists": "tag_product"}
    {"and": [f1, f2, ...]}  {"or": [...]}  {"not": f}

A dict with several operators, or several fields under one operator, is an
implicit AND. :func:`parse` turns it into a small AST; each adapter translates
the AST, and :func:`push_down_not` rewrites NOT for stores without a native NOT.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Dict, List, Mapping, Optional, Union

COMPARE = ("eq", "ne", "gt", "gte", "lt", "lte")
LIST_OPS = ("in", "nin")
BOOL_OPS = ("and", "or")
ALL_OPS = COMPARE + LIST_OPS + BOOL_OPS + ("not", "exists")
Scalar = Union[str, int, float, bool]

Where = Union[Mapping[str, Any], "Node"]


class FilterError(ValueError):
    """The filter is malformed, or the store can't express it."""


@dataclass
class Node:
    op: str
    field: Optional[str] = None
    value: Any = None
    args: List["Node"] = dc_field(default_factory=list)


def _scalar(v: Any, where: str) -> Scalar:
    if isinstance(v, (str, int, float, bool)):
        return v
    raise FilterError(f"{where}: values must be str, int, float or bool, got {type(v).__name__}")


def parse(where: Optional[Where]) -> Optional[Node]:
    if where is None or where == {}:
        return None
    if isinstance(where, Node):
        return where
    if not isinstance(where, Mapping):
        raise FilterError(f"filter must be a dict, got {type(where).__name__}")
    nodes: List[Node] = []
    for op, body in where.items():
        if op not in ALL_OPS:
            raise FilterError(f"unknown filter operator {op!r}; use one of {ALL_OPS}")
        if op in BOOL_OPS:
            if not isinstance(body, list) or not body:
                raise FilterError(f"{op!r} needs a non-empty list of filters")
            kids = [n for n in (parse(b) for b in body) if n is not None]
            nodes.append(kids[0] if len(kids) == 1 else Node(op, args=kids))
        elif op == "not":
            kid = parse(body)
            if kid is None:
                raise FilterError("'not' needs a filter")
            nodes.append(Node("not", args=[kid]))
        elif op == "exists":
            fields = [body] if isinstance(body, str) else list(body)
            nodes.extend(Node("exists", field=f) for f in fields)
        else:
            if not isinstance(body, Mapping) or not body:
                raise FilterError(f"{op!r} needs {{field: value}}")
            for f, v in body.items():
                if op in LIST_OPS:
                    if not isinstance(v, (list, tuple)) or not v:
                        raise FilterError(f"{op!r} on {f!r} needs a non-empty list")
                    nodes.append(Node(op, field=f, value=[_scalar(x, f) for x in v]))
                else:
                    nodes.append(Node(op, field=f, value=_scalar(v, f)))
    return nodes[0] if len(nodes) == 1 else Node("and", args=nodes)


def and_(*parts: Optional[Node]) -> Optional[Node]:
    kids: List[Node] = []
    for p in parts:
        if p is None:
            continue
        kids.extend(p.args if p.op == "and" else [p])
    if not kids:
        return None
    return kids[0] if len(kids) == 1 else Node("and", args=kids)


_NEGATE = {"eq": "ne", "ne": "eq", "in": "nin", "nin": "in", "gt": "lte", "gte": "lt", "lt": "gte", "lte": "gt"}


def push_down_not(node: Optional[Node], *, allow_not_exists: bool = False) -> Optional[Node]:
    """Rewrite NOT away using De Morgan and operator negation.

    ``not exists`` survives only when ``allow_not_exists`` (the store has it natively).
    Note: negated comparisons match only records that have the field.
    """
    if node is None:
        return None

    def neg(n: Node) -> Node:
        if n.op == "not":
            return rec(n.args[0])
        if n.op in _NEGATE:
            return Node(_NEGATE[n.op], field=n.field, value=n.value)
        if n.op == "and":
            return Node("or", args=[neg(a) for a in n.args])
        if n.op == "or":
            return Node("and", args=[neg(a) for a in n.args])
        if n.op == "exists":
            if allow_not_exists:
                return Node("not", args=[n])
            raise FilterError("this store can't express 'not exists'")
        raise FilterError(f"can't negate {n.op!r}")

    def rec(n: Node) -> Node:
        if n.op == "not":
            return neg(n.args[0])
        if n.op in BOOL_OPS:
            return Node(n.op, args=[rec(a) for a in n.args])
        return n

    return rec(node)


def fields_of(node: Optional[Node]) -> List[str]:
    if node is None:
        return []
    if node.field:
        return [node.field]
    out: List[str] = []
    for a in node.args:
        out.extend(fields_of(a))
    return out


_MISSING = object()


def matches(node: Optional[Node], metadata: Mapping[str, Any]) -> bool:
    """Reference evaluator: the semantics every adapter must reproduce."""
    if node is None:
        return True
    op = node.op
    if op == "and":
        return all(matches(a, metadata) for a in node.args)
    if op == "or":
        return any(matches(a, metadata) for a in node.args)
    if op == "not":
        return not matches(node.args[0], metadata)
    v = metadata.get(node.field, _MISSING) if node.field else _MISSING
    if op == "exists":
        return v is not _MISSING and v is not None
    if v is _MISSING or v is None:
        return False  # comparisons (including ne / nin) only match records that have the field
    if op == "eq":
        return _same(v, node.value)
    if op == "ne":
        return not _same(v, node.value)
    if op == "in":
        return any(_same(v, x) for x in node.value)
    if op == "nin":
        return not any(_same(v, x) for x in node.value)
    try:
        if op == "gt":
            return v > node.value
        if op == "gte":
            return v >= node.value
        if op == "lt":
            return v < node.value
        if op == "lte":
            return v <= node.value
    except TypeError:
        return False
    raise FilterError(f"unknown op {op!r}")


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    return a == b


def to_dict(node: Optional[Node]) -> Optional[Dict[str, Any]]:
    """AST back to user syntax (for traces)."""
    if node is None:
        return None
    if node.op in BOOL_OPS:
        return {node.op: [to_dict(a) for a in node.args]}
    if node.op == "not":
        return {"not": to_dict(node.args[0])}
    if node.op == "exists":
        return {"exists": node.field}
    return {node.op: {node.field: node.value}}
