"""Confidence-gated column matching: maps a new program's column headers onto
bfar's canonical items, abstaining rather than guessing.

bfar's canonical names are opaque survey codes (D1.2:A_MOTORC), so string
similarity against them is hopeless -- instead each canonical item in the
taxonomy carries a hand-curated keywords list ("motorcycle", "motorbike"...)
and matching is keyword containment against the incoming header. The design
bias is deliberate: a column that can't be matched confidently is simply left
unmapped (the request still works via tier-3 adaptation); the only way this
module can be wrong is to be *confidently* wrong, which the bijectivity rule
below narrows further.
"""
import re
from difflib import SequenceMatcher

# Tokens that mark a column as a quantity/count variant. A canonical
# quantity item (D1.2-A_QTY) only matches headers carrying one of these, and
# a flag item (D1.2:A_MOTORC) only matches headers that DON'T -- otherwise
# "num_motorcycles" would happily match the ownership flag.
_QTY_TOKENS = {"qty", "quantity", "count", "num", "number", "no", "pcs", "units"}


def _tokens(s):
    return [t for t in re.split(r"[^a-z0-9]+", str(s).lower()) if t]


def _joined(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _is_quantity_header(tokens):
    return any(t in _QTY_TOKENS or t.startswith("num") for t in tokens)


def _keyword_hit(keyword, tokens, joined):
    # Short keywords (tv, ac, cp, ref, sss...) must match a whole token, or
    # "car" would fire on "carpet". Longer compound keywords (powersupply,
    # lifeinsurance) may match as substrings of the squashed header so
    # "owns_power_supply" and "ownspowersupply" both work.
    if keyword in tokens:
        return True
    return len(keyword) >= 6 and keyword in joined


def match_columns(incoming_columns, taxonomy):
    """
    Returns (mapping, scores): mapping = {canonical_item: incoming_column}
    for confident matches only; scores = {canonical_item: score} for those.

    Resolution is bijective-or-abstain: every incoming column is claimed by
    at most one canonical item and vice versa. When two candidates tie for
    the same column (or two columns tie for the same item), BOTH are dropped
    rather than picking one arbitrarily.
    """
    candidates = []  # (score, canonical_item, incoming_column)

    for index_def in taxonomy["indices"].values():
        for item_name, item in index_def["items"].items():
            keywords = item.get("keywords", [])
            wants_qty = item.get("quantity", False)
            for col in incoming_columns:
                toks = _tokens(col)
                joined = _joined(col)
                if not joined or _is_quantity_header(toks) != wants_qty:
                    continue
                hit_lengths = [len(kw) for kw in keywords if _keyword_hit(kw, toks, joined)]
                if not hit_lengths:
                    continue
                # Specificity: how much of the header the best keyword
                # explains, with a fuzzy-similarity tiebreak on top.
                best_kw = max(hit_lengths)
                score = best_kw / max(len(joined), 1) + 0.001 * SequenceMatcher(None, _joined(item_name), joined).ratio()
                candidates.append((score, item_name, col))

    # Best canonical item per incoming column; exact ties -> abstain.
    by_column = {}
    for score, item_name, col in candidates:
        current = by_column.get(col)
        if current is None or score > current[0]:
            by_column[col] = (score, item_name, False)
        elif score == current[0] and item_name != current[1]:
            by_column[col] = (score, current[1], True)  # tied -> poisoned

    # Best incoming column per canonical item, over the survivors.
    by_item = {}
    for col, (score, item_name, tied) in by_column.items():
        if tied:
            continue
        current = by_item.get(item_name)
        if current is None or score > current[0]:
            by_item[item_name] = (score, col, False)
        elif score == current[0] and col != current[1]:
            by_item[item_name] = (score, current[1], True)

    mapping, scores = {}, {}
    for item_name, (score, col, tied) in by_item.items():
        if not tied:
            mapping[item_name] = col
            scores[item_name] = round(score, 4)
    return mapping, scores
