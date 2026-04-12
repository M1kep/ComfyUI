"""Layer-spec mini-language (v0.1 minimum grammar).

    all                  → every site
    cross | self         → attn2-only / attn1-only
    in.* | mid | out.*   → stage filter
    out.0-5              → range within stage
    out.4,out.7,mid      → comma-separated union
"""
from __future__ import annotations

import re

from .blocks import BlockId, BlockInfo

_STAGE = {"in": "input", "mid": "middle", "out": "output",
          "input": "input", "middle": "middle", "output": "output"}

GRAMMAR = __doc__


def parse_int_spec(spec: str, n: int) -> list[int]:
    """``"all" | "0-3,7"`` → sorted list of ints in ``[0, n)``."""
    spec = spec.strip()
    if spec in ("", "all", "*"):
        return list(range(n))
    out: set[int] = set()
    for part in (p.strip() for p in spec.split(",")):
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-"))
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return sorted(i for i in out if 0 <= i < n)


def _match_one(term: str, b: BlockInfo) -> bool:
    if term in ("all", "*"):
        return True
    if term == "cross":
        return b.sub == "attn2" or b.is_joint
    if term == "self":
        return b.sub == "attn1" or b.is_joint
    if term in ("attn1", "attn2"):
        return b.sub == term

    m = re.fullmatch(r"(in|mid|out|input|middle|output)(?:\.(\*|\d+(?:-\d+)?))?", term)
    if not m:
        raise ValueError(
            f"AttentionLab: cannot parse layer-spec term {term!r}.\n{GRAMMAR}")
    stage = _STAGE[m.group(1)]
    if b.stage != stage:
        return False
    sel = m.group(2)
    if sel is None or sel == "*":
        return True
    if "-" in sel:
        lo, hi = (int(x) for x in sel.split("-"))
        return lo <= b.id[1] <= hi
    return b.id[1] == int(sel)


def parse_layer_spec(spec: str, blocks: list[BlockInfo]) -> list[BlockId]:
    spec = (spec or "all").strip()
    chosen: list[BlockId] = []
    seen: set[BlockId] = set()
    for term in (t.strip() for t in spec.split(",") if t.strip()):
        # allow "<stage>.<sel> <sub-filter>" e.g. "out.4 attn2"
        parts = term.split()
        sub_filter = None
        if len(parts) == 2 and parts[1] in ("attn1", "attn2", "cross", "self"):
            term, sf = parts
            sub_filter = "attn2" if sf in ("attn2", "cross") else "attn1"
        for b in blocks:
            if sub_filter is not None and b.sub != sub_filter:
                continue
            if _match_one(term, b) and b.id not in seen:
                chosen.append(b.id)
                seen.add(b.id)
    if not chosen:
        raise ValueError(
            f"AttentionLab: layer spec {spec!r} matched no blocks "
            f"(model has {len(blocks)} sites).\n{GRAMMAR}")
    return chosen
