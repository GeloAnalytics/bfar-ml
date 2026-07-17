"""Composite-index computation for the tier-2 scoring path.

The 57 raw bfar features compress into 6 universal indices (transport assets,
household durables, connectivity, utilities access, housing, social
protection) so that programs with different column headers can still be
scored by a frozen baseline model: a mapping {canonical_item: incoming_column}
asserts which of the program's columns mean the same thing as bfar's, and
each mapped column inherits the canonical item's bfar standardization
(z-scored against bfar's mean/std) and its within-index weight (the raw
baseline model's feature importance). The index model itself never sees raw
columns, only these 6 numbers, which is what makes it portable.
"""
import json
import os

import numpy as np
import pandas as pd

INDEX_NAMES = [
    "transport_assets",
    "household_durables",
    "connectivity",
    "utilities_access",
    "housing",
    "social_protection",
]


def load_taxonomy(path):
    with open(path, "r") as f:
        return json.load(f)


def load_index_stats(path):
    with open(path, "r") as f:
        return json.load(f)


def compute_indices(df, mapping, taxonomy, index_stats=None):
    """
    Folds `df`'s columns into the 6 composite indices via `mapping`
    ({canonical_item: incoming_column}; use an identity mapping for bfar
    itself). Per index: weighted mean of the mapped items' z-scores (bfar
    mean/std from the taxonomy), weights renormalized over the items actually
    present. Missing values within a mapped column fall back to the item's
    bfar mean (z contribution 0), so sparse columns degrade gracefully
    instead of poisoning the index.

    An index with no mapped items at all is imputed with bfar's median index
    value from `index_stats` (0.0 if stats not given) and reported in the
    returned `imputed` list -- callers surface this so a consumer can see
    how much of the score rests on real data.

    Returns (index_df with INDEX_NAMES columns aligned to df.index, imputed).
    """
    out = {}
    imputed = []

    for index_name in INDEX_NAMES:
        items = taxonomy["indices"][index_name]["items"]
        weighted_sum = np.zeros(len(df), dtype=float)
        weight_total = 0.0

        for item_name, item in items.items():
            col = mapping.get(item_name)
            if col is None or col not in df.columns:
                continue
            std = item["std"] if item["std"] > 0 else 1.0
            values = pd.to_numeric(df[col], errors="coerce").fillna(item["mean"]).to_numpy(dtype=float)
            z = (values - item["mean"]) / std
            weight = item["weight"]
            weighted_sum += weight * z
            weight_total += weight

        if weight_total > 0:
            out[index_name] = weighted_sum / weight_total
        else:
            fallback = index_stats[index_name]["median"] if index_stats else 0.0
            out[index_name] = np.full(len(df), fallback, dtype=float)
            imputed.append(index_name)

    return pd.DataFrame(out, index=df.index)[INDEX_NAMES], imputed


def indices_covered(mapping, taxonomy, columns):
    """How many of the 6 indices have at least one mapped item present in
    `columns` -- the coverage number the promotion gate checks."""
    columns = set(columns)
    covered = 0
    for index_name in INDEX_NAMES:
        items = taxonomy["indices"][index_name]["items"]
        if any(mapping.get(item) in columns for item in items):
            covered += 1
    return covered
