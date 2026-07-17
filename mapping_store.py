"""Draft/registered mapping store with the automatic promotion state machine.

A "mapping" is small JSON metadata ({canonical_item: incoming_column}) --
never model weights, never uploaded data -- so the frozen-baseline guarantee
survives: registering a program teaches the service how to READ that
program's columns, not anything learned from its rows.

Promotion is automatic but deliberately conservative; a mapping only becomes
registered when all three gates pass:
  1. the matcher produced it confidently (column_matcher already abstained
     on anything ambiguous),
  2. the identical mapping showed up on PROMOTION_CONSISTENT_UPLOADS
     consecutive uploads covering >= MIN_INDICES_COVERED indices,
  3. the index values it produces on the triggering upload look like data
     the baseline has seen (|mean z| <= SANITY_MAX_ABS_MEAN per index) --
     the backstop for a mapping that matched confidently and consistently
     but is still semantically wrong.

Everything is plain JSON on disk under mappings/ (gitignored runtime state),
with an append-only audit log, so a bad auto-promotion is findable after the
fact and reversible by a single demote() (or deleting the file).
"""
import hashlib
import json
import os
import re
import time

PROMOTION_CONSISTENT_UPLOADS = 3
MIN_INDICES_COVERED = 4
SANITY_MAX_ABS_MEAN = 3.0


def column_signature(columns):
    """Stable identity for a column schema: order-independent, separator/case
    insensitive, so the same export re-uploaded always lands on the same
    draft."""
    normalized = sorted(re.sub(r"[^a-z0-9]+", "", str(c).lower()) for c in columns)
    return hashlib.sha1("|".join(normalized).encode()).hexdigest()[:12]


def safe_key(key):
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(key))[:64]


class MappingStore:
    def __init__(self, root):
        self.root = root
        self.registered_dir = os.path.join(root, "registered")
        self.drafts_dir = os.path.join(root, "drafts")
        self.audit_path = os.path.join(root, "audit.log")
        os.makedirs(self.registered_dir, exist_ok=True)
        os.makedirs(self.drafts_dir, exist_ok=True)

    # -- plumbing ---------------------------------------------------------

    def _write_json(self, path, payload):
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)

    def _read_json(self, path):
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def _audit(self, event, **fields):
        entry = {"ts": time.time(), "event": event, **fields}
        with open(self.audit_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _registered_path(self, program_key):
        return os.path.join(self.registered_dir, safe_key(program_key) + ".json")

    def _draft_path(self, program_key):
        return os.path.join(self.drafts_dir, safe_key(program_key) + ".json")

    # -- reads ------------------------------------------------------------

    def get_registered(self, program_key):
        return self._read_json(self._registered_path(program_key))

    def list_all(self):
        def load_dir(d):
            out = []
            for name in sorted(os.listdir(d)):
                if name.endswith(".json"):
                    payload = self._read_json(os.path.join(d, name))
                    if payload:
                        out.append(payload)
            return out
        return {"registered": load_dir(self.registered_dir), "drafts": load_dir(self.drafts_dir)}

    # -- the promotion state machine ---------------------------------------

    def record_observation(self, program_key, signature, mapping, coverage, index_df):
        """Called on every tier-3 request that produced a non-empty confident
        mapping. Advances (or resets) the draft for `program_key`, and
        promotes it to registered once all gates pass. Never affects the
        request that triggered it -- only future requests see a promotion.

        Returns a status dict for embedding in the API response.
        """
        now = time.time()
        draft = self._read_json(self._draft_path(program_key))

        if draft and draft.get("mapping") == mapping and draft.get("signature") == signature:
            draft["consistent_count"] += 1
            draft["last_seen"] = now
        else:
            if draft:
                self._audit("draft_reset", program_key=program_key, signature=signature,
                            previous_count=draft.get("consistent_count"))
            draft = {
                "program_key": program_key,
                "signature": signature,
                "mapping": mapping,
                "coverage": coverage,
                "consistent_count": 1,
                "first_seen": now,
                "last_seen": now,
            }
        self._write_json(self._draft_path(program_key), draft)

        status = {
            "matched_items": len(mapping),
            "indices_covered": coverage,
            "draft_consistent_count": draft["consistent_count"],
            "promoted": False,
        }

        if draft["consistent_count"] < PROMOTION_CONSISTENT_UPLOADS or coverage < MIN_INDICES_COVERED:
            return status

        # Sanity gate: indices are z-scored against bfar, so bfar itself sits
        # near 0. A covered index whose dataset-wide mean lands far outside
        # that says the mapping is reading the wrong thing, however
        # confidently the names matched.
        offending = {}
        for index_name in index_df.columns:
            mean = float(index_df[index_name].mean())
            if abs(mean) > SANITY_MAX_ABS_MEAN:
                offending[index_name] = round(mean, 3)
        if offending:
            self._audit("promotion_rejected_sanity", program_key=program_key,
                        signature=signature, offending_index_means=offending,
                        consistent_count=draft["consistent_count"])
            status["sanity_rejected"] = offending
            return status

        registered = {
            "program_key": program_key,
            "signature": signature,
            "mapping": mapping,
            "coverage": coverage,
            "source": "auto",
            "promoted_at": now,
            "uploads_seen": draft["consistent_count"],
        }
        self._write_json(self._registered_path(program_key), registered)
        os.remove(self._draft_path(program_key))
        self._audit("promoted", program_key=program_key, signature=signature,
                    matched_items=len(mapping), coverage=coverage,
                    uploads_seen=draft["consistent_count"])
        status["promoted"] = True
        return status

    def demote(self, program_key):
        """Removes a registered mapping (and any draft, so it can't instantly
        re-promote off a stale counter). Returns True if something existed."""
        removed = False
        for path in (self._registered_path(program_key), self._draft_path(program_key)):
            if os.path.exists(path):
                os.remove(path)
                removed = True
        if removed:
            self._audit("demoted", program_key=program_key)
        return removed
