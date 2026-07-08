"""Hash-chained audit ledger for federated training rounds.

A lightweight, dependency-free "blockchain-style" immutable log: each training round is
recorded as a block containing a hash of the aggregated model weights for that round, a
timestamp, optional round metrics, and the hash of the previous block. Because each
block's own hash is computed over its content plus the previous block's hash, altering
any past round's recorded weights or metrics changes that block's hash and breaks every
subsequent link in the chain -- `verify_chain` detects this deterministically.

This provides integrity and tamper-evidence for the training history (who trained what,
in what order, with what result) without requiring a distributed consensus network: a
single append-only, independently re-verifiable log is what the audit-trail requirement
actually calls for here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

GENESIS_HASH = "0" * 64


def _hash_weights(weights: List[np.ndarray]) -> str:
    """Deterministic SHA-256 hash of a list of weight arrays."""
    hasher = hashlib.sha256()
    for w in weights:
        hasher.update(np.ascontiguousarray(w, dtype=np.float64).tobytes())
    return hasher.hexdigest()


@dataclass
class Block:
    index: int
    round_number: int
    timestamp: float
    weights_hash: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    previous_hash: str = GENESIS_HASH
    block_hash: str = ""

    def compute_hash(self) -> str:
        payload = {
            "index": self.index,
            "round_number": self.round_number,
            "timestamp": self.timestamp,
            "weights_hash": self.weights_hash,
            "metrics": self.metrics,
            "previous_hash": self.previous_hash,
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class AuditLedger:
    """Append-only, hash-chained log of federated training rounds."""

    def __init__(self) -> None:
        self.chain: List[Block] = []

    def append_round(self, round_number: int, weights: List[np.ndarray], metrics: Optional[Dict[str, Any]] = None) -> Block:
        previous_hash = self.chain[-1].block_hash if self.chain else GENESIS_HASH
        block = Block(
            index=len(self.chain),
            round_number=round_number,
            timestamp=time.time(),
            weights_hash=_hash_weights(weights),
            metrics=metrics or {},
            previous_hash=previous_hash,
        )
        block.block_hash = block.compute_hash()
        self.chain.append(block)
        return block

    def verify_chain(self) -> Tuple[bool, Optional[int]]:
        """Recompute every block's hash and check the links; returns (valid, first_bad_index)."""
        expected_previous = GENESIS_HASH
        for block in self.chain:
            if block.previous_hash != expected_previous:
                return False, block.index
            if block.compute_hash() != block.block_hash:
                return False, block.index
            expected_previous = block.block_hash
        return True, None

    def to_dict(self) -> List[Dict[str, Any]]:
        return [asdict(b) for b in self.chain]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "AuditLedger":
        ledger = cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        ledger.chain = [Block(**b) for b in data]
        return ledger
